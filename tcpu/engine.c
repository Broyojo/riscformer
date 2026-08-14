/* Fast execution engine for the constructed transformer CPU.
 *
 * Computes the identical forward pass as tcpu/nn.py — multi-head softmax
 * attention + ReLU MLP with residual stream — but stores the (extremely
 * sparse) constructed weights in CSR form and tracks which residual dims are
 * nonzero, so each step costs ~10^5 FLOPs instead of ~10^9. Same weights,
 * same math, same result (bit-exact architectural state; activations saturate
 * to exact 0/1 by construction).
 *
 * Also hosts the memory bus (flat RAM) and the step loop, returning to the
 * host process only for ecall/ebreak (semihosting).
 *
 * cc -O2 -shared -o libtcpu.dylib engine.c
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdatomic.h>

#ifdef PROFILE
static double prof_ns[8];
static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#define PBEG(x) double _p##x = now_ns();
#define PEND(x, i) prof_ns[i] += now_ns() - _p##x;
void tf_prof(double *dst) { for (int i = 0; i < 8; i++) dst[i] = prof_ns[i]; }
void tf_prof_reset(void) { memset(prof_ns, 0, sizeof prof_ns); }
#else
#define PBEG(x)
#define PEND(x, i)
void tf_prof(double *dst) { for (int i = 0; i < 8; i++) dst[i] = 0; }
void tf_prof_reset(void) {}
#endif

#define T 34
#define T_CTL 0
#define T_INSTR 1
#define T_REG0 2
#define MAXH 8192

typedef struct {
    int rows, cols, nnz;
    int *rowptr, *col; float *val;
    int n_ne; int *ne_rows;     /* rows with at least one entry */
    uint64_t *rowbits;          /* bitmap: row has entries */
} Sp;
typedef struct { Sp Wq, Wk, Wv, Wo; } Head;
typedef struct {
    int nheads; Head *heads;
    int H; Sp W1, W2; float *b1;
    int nbase; int *base_idx; float *base_val;  /* relu(b1) @ W2, precomputed */
} Block;

static int D, NB;
static Block *blocks;

/* dim table */
typedef struct { char name[32]; int base, size; } Field;
static Field *fields; static int nfields;
static int f_TT_CTL, f_TT_INSTR, f_TT_REG, f_RIDX, f_VAL, f_IW, f_PC, f_PLF,
    f_PLRD, f_PLD, f_NPC, f_F3, f_AS, f_BRAW, f_PLF_N, f_PLRD_N, f_IMM, f_ISM;
static int f_CLS[16]; static int nCLS;
static int f_CLS_LOAD, f_CLS_STORE, f_CLS_SYS;

/* machine state */
static uint8_t *ram; static uint64_t ram_base, ram_size;
static uint32_t pc, regs[32];
static int plf; static uint32_t plrd, pld;
static uint64_t steps;
static int val_seeded;   /* reg VAL bits live in x across steps once seeded */

/* forward buffers */
static float *x;                 /* T x D */
static int *nzidx; static int *nzcnt; static uint8_t *nzmask; /* per row */
static float qb[T * 64], kb[T * 64], vb[T * 64], ob[T * 64];
static float scores[T][T];
static __thread float acc[MAXH];
static __thread int touched[MAXH];
static __thread uint8_t tmask[MAXH];

/* ---- thread pool: SPMD over tokens, spin barriers ---- */
#define MAXTHR 8
static int nthreads = 1;
static int owner[T];
static pthread_t pool[MAXTHR];
static _Atomic int bar_count, bar_gen;
static _Atomic int pool_go;       /* -1 = shutdown */
static uint32_t cur_inst;

static void barrier_wait(void) {
    int gen = atomic_load_explicit(&bar_gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&bar_count, 1, memory_order_acq_rel)
            == nthreads - 1) {
        atomic_store_explicit(&bar_count, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&bar_gen, 1, memory_order_release);
    } else {
        int spins = 0;
        while (atomic_load_explicit(&bar_gen, memory_order_acquire) == gen)
            if (++spins > 100000) { sched_yield(); spins = 0; }
    }
}
static void forward_worker(int tid);
static void *worker_main(void *arg) {
    int tid = (int)(intptr_t)arg;
    for (;;) {
        barrier_wait();                       /* step start */
        if (atomic_load(&pool_go) < 0) return 0;
        forward_worker(tid);
        barrier_wait();                       /* step end */
    }
}
static void pool_init(void) {
    static int inited = 0;
    if (inited) return;
    inited = 1;
    const char *e = getenv("TCPU_THREADS");
    int n = e ? atoi(e) : 4;
    if (n < 1) n = 1;
    if (n > MAXTHR) n = MAXTHR;
    nthreads = n;
    for (int t = 0; t < T; t++)
        owner[t] = (nthreads == 1) ? 0
                 : (t == T_INSTR ? 0 : 1 + (t % (nthreads - 1)));
    for (int i = 1; i < nthreads; i++)
        pthread_create(&pool[i], 0, worker_main, (void *)(intptr_t)i);
}

static int lookup(const char *n) {
    for (int i = 0; i < nfields; i++)
        if (!strcmp(fields[i].name, n)) return i;
    fprintf(stderr, "tcpu: missing field %s\n", n); exit(1);
}
static int fbase(const char *n) { return fields[lookup(n)].base; }

static void read_sp(FILE *f, Sp *m) {
    int hdr[3];
    fread(hdr, 4, 3, f);
    m->rows = hdr[0]; m->cols = hdr[1]; m->nnz = hdr[2];
    m->rowptr = malloc(4 * (m->rows + 1));
    m->col = malloc(4 * (m->nnz > 0 ? m->nnz : 1));
    m->val = malloc(4 * (m->nnz > 0 ? m->nnz : 1));
    fread(m->rowptr, 4, m->rows + 1, f);
    fread(m->col, 4, m->nnz, f);
    fread(m->val, 4, m->nnz, f);
    m->n_ne = 0;
    m->ne_rows = malloc(4 * (m->rows > 0 ? m->rows : 1));
    m->rowbits = calloc((m->rows >> 6) + 1, 8);
    for (int r = 0; r < m->rows; r++)
        if (m->rowptr[r + 1] > m->rowptr[r]) {
            m->ne_rows[m->n_ne++] = r;
            m->rowbits[r >> 6] |= 1ull << (r & 63);
        }
}

int tf_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8]; fread(magic, 1, 8, f);
    if (memcmp(magic, "TCPU1", 5)) return -2;
    int hdr[2]; fread(hdr, 4, 2, f);
    D = hdr[0]; NB = hdr[1];
    fread(&nfields, 4, 1, f);
    fields = calloc(nfields, sizeof(Field));
    for (int i = 0; i < nfields; i++) {
        int nl; fread(&nl, 4, 1, f);
        fread(fields[i].name, 1, nl, f);
        int bs[2]; fread(bs, 4, 2, f);
        fields[i].base = bs[0]; fields[i].size = bs[1];
    }
    blocks = calloc(NB, sizeof(Block));
    for (int b = 0; b < NB; b++) {
        Block *B = &blocks[b];
        fread(&B->nheads, 4, 1, f);
        B->heads = calloc(B->nheads > 0 ? B->nheads : 1, sizeof(Head));
        for (int h = 0; h < B->nheads; h++) {
            read_sp(f, &B->heads[h].Wq); read_sp(f, &B->heads[h].Wk);
            read_sp(f, &B->heads[h].Wv); read_sp(f, &B->heads[h].Wo);
        }
        fread(&B->H, 4, 1, f);
        if (B->H) {
            if (B->H > MAXH) { fprintf(stderr, "H too big\n"); exit(1); }
            read_sp(f, &B->W1);
            B->b1 = malloc(4 * B->H);
            fread(B->b1, 4, B->H, f);
            read_sp(f, &B->W2);
            /* precompute relu(b1) @ W2 */
            float *dense = calloc(D, 4);
            for (int h = 0; h < B->H; h++) {
                float v = B->b1[h] > 0 ? B->b1[h] : 0;
                if (v == 0) continue;
                for (int e = B->W2.rowptr[h]; e < B->W2.rowptr[h + 1]; e++)
                    dense[B->W2.col[e]] += v * B->W2.val[e];
            }
            B->nbase = 0;
            for (int d = 0; d < D; d++) if (dense[d] != 0) B->nbase++;
            B->base_idx = malloc(4 * (B->nbase ? B->nbase : 1));
            B->base_val = malloc(4 * (B->nbase ? B->nbase : 1));
            int j = 0;
            for (int d = 0; d < D; d++) if (dense[d] != 0) {
                B->base_idx[j] = d; B->base_val[j++] = dense[d];
            }
            free(dense);
        }
    }
    fclose(f);
    /* resolve fields */
    f_TT_CTL = fbase("TT_CTL"); f_TT_INSTR = fbase("TT_INSTR");
    f_TT_REG = fbase("TT_REG"); f_RIDX = fbase("RIDX"); f_VAL = fbase("VAL");
    f_IW = fbase("IW"); f_PC = fbase("PC"); f_PLF = fbase("PLF");
    f_PLRD = fbase("PLRD"); f_PLD = fbase("PLD"); f_NPC = fbase("NPC");
    f_F3 = fbase("F3"); f_AS = fbase("AS"); f_BRAW = fbase("BRAW");
    f_PLF_N = fbase("PLF_N"); f_PLRD_N = fbase("PLRD_N"); f_IMM = fbase("IMM");
    f_CLS_LOAD = fbase("CLS_LOAD"); f_CLS_STORE = fbase("CLS_STORE");
    f_CLS_SYS = fbase("CLS_SYS"); f_ISM = fbase("ISM");
    nCLS = 0;
    for (int i = 0; i < nfields; i++)
        if (!strncmp(fields[i].name, "CLS_", 4)) f_CLS[nCLS++] = fields[i].base;
    x = calloc((size_t)T * D, 4);
    nzidx = malloc(4 * (size_t)T * D);
    nzcnt = calloc(T, 4);
    nzmask = calloc((size_t)T * D, 1);
    pool_init();
    return 0;
}

int tf_init_ram(uint64_t base, uint64_t size) {
    ram_base = base; ram_size = size;
    ram = calloc(size, 1);
    return ram ? 0 : -1;
}
uint8_t *tf_ram(void) { return ram; }
uint64_t tf_ram_base(void) { return ram_base; }

void tf_reset(uint32_t pc0) {
    pc = pc0; memset(regs, 0, sizeof regs);
    plf = 0; steps = 0; val_seeded = 0;
}

static inline uint32_t val_from_x(int r);
static void sync_regs_from_x(void) {
    if (!val_seeded) return;
    for (int r = 0; r < 32; r++) regs[r] = val_from_x(r);
}
uint32_t tf_pc(void) { return pc; }
uint64_t tf_steps(void) { return steps; }
uint32_t tf_getreg(int r) {
    if (plf && (int)plrd == r) return pld;
    return val_seeded ? val_from_x(r) : regs[r];
}
void tf_setreg(int r, uint32_t v) {
    if (plf && (int)plrd == r) { pld = v; return; }
    regs[r] = v;
    if (val_seeded) {   /* mirror into the live residual bits */
        float *row = x + (size_t)(T_REG0 + r) * D;
        uint8_t *m = nzmask + (size_t)(T_REG0 + r) * D;
        for (int i = 0; i < 32; i++) {
            int d = f_VAL + i;
            float bit = (float)((v >> i) & 1);
            if (row[d] != bit) {
                if (!m[d]) { m[d] = 1;
                    nzidx[(T_REG0 + r) * D + nzcnt[T_REG0 + r]++] = d; }
                row[d] = bit;
            }
        }
    }
}

static uint32_t bus_read(uint32_t a, int size, int sext_) {
    uint64_t off = (uint64_t)a - ram_base;
    if (off + size > ram_size) return 0;
    uint32_t v = 0;
    memcpy(&v, ram + off, size);
    if (sext_) {
        int sh = 32 - 8 * size;
        v = (uint32_t)(((int32_t)(v << sh)) >> sh);
    }
    return v;
}
static void bus_write(uint32_t a, int size, uint32_t v) {
    uint64_t off = (uint64_t)a - ram_base;
    if (off + size > ram_size) return;
    memcpy(ram + off, &v, size);
}

static inline void setx(int t, int d, float v) {
    float *row = x + (size_t)t * D;
    uint8_t *m = nzmask + (size_t)t * D;
    row[d] += v;
    if (!m[d]) { m[d] = 1; nzidx[t * D + nzcnt[t]++] = d; }
}

static void embed(uint32_t inst) {
    /* clear dims dirtied last step -- except register VAL bits, which persist
     * in place (they are the architectural state; exact 0/1 floats) */
    for (int t = 0; t < T; t++) {
        float *row = x + (size_t)t * D;
        uint8_t *m = nzmask + (size_t)t * D;
        int *nz = nzidx + t * D;
        if (val_seeded && t >= T_REG0) {
            int keep = 0;
            for (int i = 0; i < nzcnt[t]; i++) {
                int d = nz[i];
                if (d >= f_VAL && d < f_VAL + 32) { nz[keep++] = d; continue; }
                row[d] = 0.0f; m[d] = 0;
            }
            nzcnt[t] = keep;
            continue;
        }
        for (int i = 0; i < nzcnt[t]; i++) {
            row[nz[i]] = 0.0f;
            m[nz[i]] = 0;
        }
        nzcnt[t] = 0;
    }
    setx(T_CTL, f_TT_CTL, 1.0f);
    for (int i = 0; i < 32; i++)
        if ((pc >> i) & 1) setx(T_CTL, f_PC + i, 1.0f);
    if (plf) {
        setx(T_CTL, f_PLF, 1.0f);
        setx(T_CTL, f_PLRD + plrd, 1.0f);
        for (int i = 0; i < 32; i++)
            if ((pld >> i) & 1) setx(T_CTL, f_PLD + i, 1.0f);
    }
    setx(T_INSTR, f_TT_INSTR, 1.0f);
    for (int i = 0; i < 32; i++)
        if ((inst >> i) & 1) setx(T_INSTR, f_IW + i, 1.0f);
    for (int r = 0; r < 32; r++) {
        int t = T_REG0 + r;
        setx(t, f_TT_REG, 1.0f);
        setx(t, f_RIDX + r, 1.0f);
        if (!val_seeded)
            for (int i = 0; i < 32; i++)
                if ((regs[r] >> i) & 1) setx(t, f_VAL + i, 1.0f);
    }
    val_seeded = 1;
}

static void sp_apply_rows(const Sp *W, const float *row, const int *nz, int cnt,
                          float *out) {
    /* out[c] += sum over nonzero input dims: row[d] * W[d, c].
     * Iterate whichever side is smaller: the matrix's nonempty rows or the
     * activation's nonzero dims (bitmap-filtered). */
    if (W->n_ne <= cnt) {
        for (int i = 0; i < W->n_ne; i++) {
            int d = W->ne_rows[i];
            float v = row[d];
            if (v == 0) continue;
            for (int e = W->rowptr[d]; e < W->rowptr[d + 1]; e++)
                out[W->col[e]] += v * W->val[e];
        }
    } else {
        const uint64_t *bits = W->rowbits;
        for (int i = 0; i < cnt; i++) {
            int d = nz[i];
            if (!((bits[d >> 6] >> (d & 63)) & 1)) continue;
            float v = row[d];
            if (v == 0) continue;
            for (int e = W->rowptr[d]; e < W->rowptr[d + 1]; e++)
                out[W->col[e]] += v * W->val[e];
        }
    }
}

static void forward_worker(int tid) {
    for (int b = 0; b < NB; b++) {
        Block *B = &blocks[b];
        for (int h = 0; h < B->nheads; h++) {
            Head *H = &B->heads[h];
            int dk = H->Wq.cols, dv = H->Wv.cols;
            PBEG(a)
            if (nthreads > 1) barrier_wait();   /* prior writes visible */
            for (int t = 0; t < T; t++) {
                if (owner[t] != tid) continue;
                memset(qb + t * dk, 0, 4 * dk); memset(kb + t * dk, 0, 4 * dk);
                memset(vb + t * dv, 0, 4 * dv); memset(ob + t * dv, 0, 4 * dv);
                const float *row = x + (size_t)t * D;
                const int *nz = nzidx + t * D;
                sp_apply_rows(&H->Wq, row, nz, nzcnt[t], qb + t * dk);
                sp_apply_rows(&H->Wk, row, nz, nzcnt[t], kb + t * dk);
                sp_apply_rows(&H->Wv, row, nz, nzcnt[t], vb + t * dv);
            }
            if (nthreads > 1) barrier_wait();   /* kb/vb complete */
            PEND(a, 1)
            PBEG(b)
            /* rows with an all-zero query attend uniformly: out = mean(v) */
            float vmean[64];
            for (int j = 0; j < dv; j++) {
                float s = 0;
                for (int u = 0; u < T; u++) s += vb[u * dv + j];
                vmean[j] = s / (float)T;
            }
            for (int t = 0; t < T; t++) {
                if (owner[t] != tid) continue;
                int qz = 1;
                for (int c = 0; c < dk; c++)
                    if (qb[t * dk + c] != 0) { qz = 0; break; }
                if (qz) {
                    memcpy(ob + t * dv, vmean, 4 * dv);
                    continue;
                }
                float mx = -1e30f;
                for (int u = 0; u < T; u++) {
                    float s = 0;
                    for (int c = 0; c < dk; c++) s += qb[t * dk + c] * kb[u * dk + c];
                    scores[t][u] = s;
                    if (s > mx) mx = s;
                }
                /* terms 30+ below the max carry weight < 1e-13 -- far under
                 * the step-gate noise margin; drop them before the expf */
                float sum = 0;
                for (int u = 0; u < T; u++) {
                    float d_ = scores[t][u] - mx;
                    float e = d_ > -30.0f ? expf(d_) : 0.0f;
                    scores[t][u] = e; sum += e;
                }
                float inv = 1.0f / sum;
                for (int u = 0; u < T; u++) {
                    float w = scores[t][u] * inv;
                    if (w < 1e-12f) continue;
                    for (int j = 0; j < dv; j++)
                        ob[t * dv + j] += w * vb[u * dv + j];
                }
            }
            PEND(b, 2)
            PBEG(c)
            for (int t = 0; t < T; t++) {
                if (owner[t] != tid) continue;
                for (int j = 0; j < dv; j++) {
                    float v = ob[t * dv + j];
                    if (v == 0) continue;
                    for (int e = H->Wo.rowptr[j]; e < H->Wo.rowptr[j + 1]; e++)
                        setx(t, H->Wo.col[e], v * H->Wo.val[e]);
                }
            }
            PEND(c, 3)
        }
        if (B->H) {
            for (int t = 0; t < T; t++) {
                if (owner[t] != tid) continue;
                const float *row = x + (size_t)t * D;
                int ntouched = 0;
                const int *nz = nzidx + t * D;
                int cnt = nzcnt[t];
                PBEG(d)
                const uint64_t *w1bits = B->W1.rowbits;
                for (int i = 0; i < cnt; i++) {
                    int d = nz[i];
                    if (!((w1bits[d >> 6] >> (d & 63)) & 1)) continue;
                    float v = row[d];
                    if (v == 0) continue;
                    for (int e = B->W1.rowptr[d]; e < B->W1.rowptr[d + 1]; e++) {
                        int hh = B->W1.col[e];
                        if (!tmask[hh]) { tmask[hh] = 1; touched[ntouched++] = hh; acc[hh] = 0; }
                        acc[hh] += v * B->W1.val[e];
                    }
                }
                PEND(d, 4)
                PBEG(e)
                for (int i = 0; i < B->nbase; i++)
                    setx(t, B->base_idx[i], B->base_val[i]);
                for (int i = 0; i < ntouched; i++) {
                    int hh = touched[i];
                    tmask[hh] = 0;
                    float b1 = B->b1[hh];
                    float h1 = acc[hh] + b1; if (h1 < 0) h1 = 0;
                    float h0 = b1 > 0 ? b1 : 0;
                    float delta = h1 - h0;
                    if (delta == 0) continue;
                    for (int e = B->W2.rowptr[hh]; e < B->W2.rowptr[hh + 1]; e++)
                        setx(t, B->W2.col[e], delta * B->W2.val[e]);
                }
                PEND(e, 5)
            }
        }
    }
}

static inline uint32_t field_u32(int t, int base, int size) {
    const float *row_ = x + (size_t)t * D;
    (void)row_;
    const float *row = x + (size_t)t * D;
    uint32_t v = 0;
    for (int i = 0; i < size; i++)
        if (row[base + i] > 0.5f) v |= 1u << i;
    return v;
}
static inline int flag(int t, int d) { return x[(size_t)t * D + d] > 0.5f; }
static inline uint32_t val_from_x(int r) {
    return field_u32(T_REG0 + r, f_VAL, 32);
}

/* return codes */
#define R_STEPS 0
#define R_ECALL 1
#define R_EBREAK 2
#define R_ILLEGAL 3

int tf_run(int64_t max_steps) {
    for (int64_t n = 0; n < max_steps; n++) {
        uint32_t inst = bus_read(pc, 4, 0);
        PBEG(f)
        embed(inst);
        PEND(f, 0)
        if (nthreads > 1) {
            barrier_wait();          /* release workers into the step */
            forward_worker(0);
            barrier_wait();          /* all writes complete */
        } else {
            forward_worker(0);
        }
        uint32_t npc = field_u32(T_INSTR, f_NPC, 32);
        uint32_t f3v = field_u32(T_INSTR, f_F3, 8);
        int f3 = 0;
        while (f3v > 1) { f3v >>= 1; f3++; }
        int any = 0;
        for (int i = 0; i < nCLS; i++) any |= flag(T_INSTR, f_CLS[i]);
        plf = 0;
        int ret = -1;
        if (!any) ret = R_ILLEGAL;
        else if (flag(T_INSTR, f_ISM) && f3 >= 4) ret = R_ILLEGAL;  /* div: -mno-div */
        else if (flag(T_INSTR, f_CLS_STORE)) {
            bus_write(field_u32(T_INSTR, f_AS, 32), 1 << (f3 & 3),
                      field_u32(T_INSTR, f_BRAW, 32));
        } else if (flag(T_INSTR, f_CLS_LOAD)) {
            uint32_t addr = field_u32(T_INSTR, f_AS, 32);
            uint32_t data = bus_read(addr, 1 << (f3 & 3), f3 < 4);
            if (flag(T_INSTR, f_PLF_N)) {
                uint32_t oh = field_u32(T_INSTR, f_PLRD_N, 32);
                plrd = 0;
                while (oh > 1) { oh >>= 1; plrd++; }
                plf = 1; pld = data;
            }
        } else if (flag(T_INSTR, f_CLS_SYS)) {
            ret = (field_u32(T_INSTR, f_IMM, 32) & 1) ? R_EBREAK : R_ECALL;
        }
        pc = npc;
        steps++;
        if (ret >= 0) return ret;
    }
    return R_STEPS;
}
