/* Minimal libc for the transformer CPU Doom port.
 * File/console IO goes through the ecall bridge (see syscalls.c). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdint.h>

int errno;

/* syscalls.c */
int _open(const char *path, int flags, int mode);
int _close(int fd);
long _read(int fd, void *buf, size_t n);
long _write(int fd, const void *buf, size_t n);
long _lseek(int fd, long off, int whence);
void _exit(int code) __attribute__((noreturn));
void *_sbrk(long incr);

/* ---------- stdio ---------- */

static FILE __stdin = {0}, __stdout = {1}, __stderr = {2};
FILE *stdin = &__stdin, *stdout = &__stdout, *stderr = &__stderr;

FILE *fopen(const char *path, const char *mode) {
    int flags = 0;
    if (mode[0] == 'w')
        flags = 1 | 0x40 | 0x200; /* O_WRONLY|O_CREAT|O_TRUNC */
    int fd = _open(path, flags, 0644);
    if (fd < 0)
        return 0;
    FILE *f = malloc(sizeof(FILE));
    f->fd = fd;
    return f;
}
int fclose(FILE *f) { int r = _close(f->fd); free(f); return r; }
size_t fread(void *p, size_t sz, size_t n, FILE *f) {
    if (!sz || !n) return 0;
    long r = _read(f->fd, p, sz * n);
    return r <= 0 ? 0 : (size_t)r / sz;
}
size_t fwrite(const void *p, size_t sz, size_t n, FILE *f) {
    if (!sz || !n) return 0;
    long r = _write(f->fd, p, sz * n);
    return r <= 0 ? 0 : (size_t)r / sz;
}
int fseek(FILE *f, long off, int whence) {
    return _lseek(f->fd, off, whence) < 0 ? -1 : 0;
}
long ftell(FILE *f) { return _lseek(f->fd, 0, 1); }
int fflush(FILE *f) { (void)f; return 0; }
int fgetc(FILE *f) {
    unsigned char c;
    return _read(f->fd, &c, 1) == 1 ? c : EOF;
}
int feof(FILE *f) { (void)f; return 0; }
int remove(const char *p) { (void)p; return -1; }
int rename(const char *a, const char *b) { (void)a; (void)b; return -1; }
int fputc(int c, FILE *f) {
    unsigned char ch = (unsigned char)c;
    return _write(f->fd, &ch, 1) == 1 ? c : EOF;
}
int putchar(int c) { return fputc(c, stdout); }
int fputs(const char *s, FILE *f) {
    size_t n = strlen(s);
    return _write(f->fd, s, n) == (long)n ? 0 : EOF;
}
int puts(const char *s) { fputs(s, stdout); return putchar('\n'); }

/* ---- formatted output ---- */

typedef struct { char *buf; size_t cap, len; FILE *f; } Sink;

static void emit(Sink *s, char c) {
    if (s->f) {
        fputc(c, s->f); /* unbuffered console/file */
    } else if (s->len + 1 < s->cap) {
        s->buf[s->len] = c;
    }
    s->len++;
}

static void emit_num(Sink *s, unsigned long long v, int base, int upper,
                     int neg, int width, int zero, int left, int prec) {
    char tmp[32];
    int n = 0;
    const char *dig = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    if (!v) tmp[n++] = '0';
    while (v) { tmp[n++] = dig[v % base]; v /= base; }
    while (prec > 0 && n < prec && n < 31) tmp[n++] = '0';
    if (neg) tmp[n++] = '-';
    int pad = width - n;
    if (!left)
        for (; pad > 0; pad--) emit(s, zero ? '0' : ' ');
    while (n) emit(s, tmp[--n]);
    if (left)
        for (; pad > 0; pad--) emit(s, ' ');
}

static void vformat(Sink *s, const char *fmt, va_list ap) {
    for (; *fmt; fmt++) {
        if (*fmt != '%') { emit(s, *fmt); continue; }
        fmt++;
        int left = 0, zero = 0, width = 0, prec = -1, ll = 0;
        for (;; fmt++) {
            if (*fmt == '-') left = 1;
            else if (*fmt == '0') zero = 1;
            else if (*fmt == '+' || *fmt == ' ' || *fmt == '#') ;
            else break;
        }
        if (*fmt == '*') { width = va_arg(ap, int); fmt++; }
        else while (isdigit(*fmt)) width = width * 10 + (*fmt++ - '0');
        if (*fmt == '.') {
            fmt++; prec = 0;
            if (*fmt == '*') { prec = va_arg(ap, int); fmt++; }
            else while (isdigit(*fmt)) prec = prec * 10 + (*fmt++ - '0');
        }
        while (*fmt == 'l') { ll++; fmt++; }
        while (*fmt == 'h' || *fmt == 'z') fmt++;
        switch (*fmt) {
        case 'd': case 'i': {
            long long v = ll > 1 ? va_arg(ap, long long)
                        : ll ? va_arg(ap, long) : va_arg(ap, int);
            int neg = v < 0;
            emit_num(s, neg ? -(unsigned long long)v : (unsigned long long)v,
                     10, 0, neg, width, zero, left, prec);
            break;
        }
        case 'u': case 'x': case 'X': case 'o': {
            unsigned long long v = ll > 1 ? va_arg(ap, unsigned long long)
                        : ll ? va_arg(ap, unsigned long) : va_arg(ap, unsigned);
            int base = *fmt == 'u' ? 10 : *fmt == 'o' ? 8 : 16;
            emit_num(s, v, base, *fmt == 'X', 0, width, zero, left, prec);
            break;
        }
        case 'p': {
            emit(s, '0'); emit(s, 'x');
            emit_num(s, (uintptr_t)va_arg(ap, void *), 16, 0, 0, 8, 1, 0, -1);
            break;
        }
        case 'c': emit(s, (char)va_arg(ap, int)); break;
        case 's': {
            const char *str = va_arg(ap, const char *);
            if (!str) str = "(null)";
            int n = 0;
            const char *q = str;
            while (*q && (prec < 0 || n < prec)) { q++; n++; }
            int pad = width - n;
            if (!left) for (; pad > 0; pad--) emit(s, ' ');
            for (int i = 0; i < n; i++) emit(s, str[i]);
            if (left) for (; pad > 0; pad--) emit(s, ' ');
            break;
        }
        case 'f': case 'g': {
            double d = va_arg(ap, double);
            if (d < 0) { emit(s, '-'); d = -d; }
            unsigned long long ip = (unsigned long long)d;
            emit_num(s, ip, 10, 0, 0, 0, 0, 0, -1);
            emit(s, '.');
            int digits = prec < 0 ? 6 : prec;
            double frac = d - (double)ip;
            for (int i = 0; i < digits; i++) {
                frac *= 10;
                int dg = (int)frac;
                emit(s, (char)('0' + dg));
                frac -= dg;
            }
            break;
        }
        case '%': emit(s, '%'); break;
        default: emit(s, '%'); emit(s, *fmt); break;
        }
    }
}

int vsnprintf(char *buf, size_t n, const char *fmt, va_list ap) {
    Sink s = {buf, n, 0, 0};
    vformat(&s, fmt, ap);
    if (n) buf[s.len < n ? s.len : n - 1] = 0;
    return (int)s.len;
}
int snprintf(char *buf, size_t n, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int r = vsnprintf(buf, n, fmt, ap);
    va_end(ap); return r;
}
int sprintf(char *buf, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int r = vsnprintf(buf, 0x7FFFFFF, fmt, ap);
    va_end(ap); return r;
}
int vfprintf(FILE *f, const char *fmt, va_list ap) {
    char buf[1024];
    Sink s = {buf, sizeof buf, 0, 0};
    vformat(&s, fmt, ap);
    size_t n = s.len < sizeof buf - 1 ? s.len : sizeof buf - 1;
    _write(f->fd, buf, n);
    return (int)s.len;
}
int fprintf(FILE *f, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int r = vfprintf(f, fmt, ap);
    va_end(ap); return r;
}
int printf(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int r = vfprintf(stdout, fmt, ap);
    va_end(ap); return r;
}

/* ---- minimal sscanf: whitespace, literals, %d %i %u %x %o %c %s ---- */
int sscanf(const char *s, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    int matched = 0;
    while (*fmt) {
        if (isspace(*fmt)) {
            while (isspace(*s)) s++;
            fmt++;
        } else if (*fmt == '%') {
            fmt++;
            while (isdigit(*fmt)) fmt++;   /* ignore width */
            int base = 10, conv = *fmt++;
            if (conv == 'i') base = 0;
            else if (conv == 'x') base = 16;
            else if (conv == 'o') base = 8;
            if (conv == 'd' || conv == 'i' || conv == 'u' || conv == 'x' || conv == 'o') {
                while (isspace(*s)) s++;
                char *end;
                long v = strtol(s, &end, base);
                if (end == s) break;
                *va_arg(ap, int *) = (int)v;
                s = end;
                matched++;
            } else if (conv == 'c') {
                if (!*s) break;
                *va_arg(ap, char *) = *s++;
                matched++;
            } else if (conv == 's') {
                while (isspace(*s)) s++;
                char *out = va_arg(ap, char *);
                if (!*s) break;
                while (*s && !isspace(*s)) *out++ = *s++;
                *out = 0;
                matched++;
            } else break;
        } else {
            if (*s != *fmt) break;
            s++; fmt++;
        }
    }
    va_end(ap);
    return matched;
}

/* ---------- stdlib ---------- */

/* bump allocator with size headers; free is a no-op (Doom's zone allocator
 * does the real memory management on top of one big malloc) */
void *malloc(size_t n) {
    n = (n + 15) & ~(size_t)15;
    char *p = _sbrk((long)(n + 16));
    if (p == (char *)-1) return 0;
    *(size_t *)p = n;
    return p + 16;
}
void free(void *p) { (void)p; }
void *calloc(size_t n, size_t sz) {
    void *p = malloc(n * sz);
    if (p) memset(p, 0, n * sz);
    return p;
}
void *realloc(void *p, size_t n) {
    if (!p) return malloc(n);
    size_t old = *(size_t *)((char *)p - 16);
    if (n <= old) return p;
    void *q = malloc(n);
    if (q) memcpy(q, p, old);
    return q;
}

void exit(int code) { _exit(code); }
void abort(void) { _write(2, "abort()\n", 8); _exit(134); }
void __assert_fail(const char *expr, const char *file, int line) {
    fprintf(stderr, "assert failed: %s at %s:%d\n", expr, file, line);
    _exit(134);
}
int abs(int x) { return x < 0 ? -x : x; }
long labs(long x) { return x < 0 ? -x : x; }
char *getenv(const char *n) { (void)n; return 0; }
int system(const char *cmd) { (void)cmd; return -1; }
int mkdir(const char *p, unsigned m) { (void)p; (void)m; return -1; }
int unlink(const char *p) { (void)p; return -1; }
int gettimeofday(void *tv, void *tz) { (void)tv; (void)tz; return -1; }

static unsigned rand_state = 1;
int rand(void) { rand_state = rand_state * 1103515245 + 12345; return (int)(rand_state >> 1); }
void srand(unsigned s) { rand_state = s; }

long strtol(const char *s, char **end, int base) {
    while (isspace(*s)) s++;
    int neg = 0;
    if (*s == '+' || *s == '-') neg = *s++ == '-';
    if ((base == 0 || base == 16) && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        base = 16; s += 2;
    } else if (base == 0 && s[0] == '0' && isdigit(s[1])) {
        base = 8; s++;
    } else if (base == 0) base = 10;
    long v = 0;
    const char *start = s;
    for (;; s++) {
        int d;
        if (isdigit(*s)) d = *s - '0';
        else if (*s >= 'a' && *s <= 'f') d = *s - 'a' + 10;
        else if (*s >= 'A' && *s <= 'F') d = *s - 'A' + 10;
        else break;
        if (d >= base) break;
        v = v * base + d;
    }
    if (end) *end = (char *)(s == start ? s : s);
    if (end && s == start) *end = (char *)start;
    return neg ? -v : v;
}
int atoi(const char *s) { return (int)strtol(s, 0, 10); }
double atof(const char *s) {
    while (isspace(*s)) s++;
    int neg = 0;
    if (*s == '+' || *s == '-') neg = *s++ == '-';
    double v = 0;
    while (isdigit(*s)) v = v * 10 + (*s++ - '0');
    if (*s == '.') {
        s++;
        double sc = 0.1;
        while (isdigit(*s)) { v += (*s++ - '0') * sc; sc *= 0.1; }
    }
    return neg ? -v : v;
}

/* ---------- string ---------- */

void *memcpy(void *dv, const void *sv, size_t n) {
    char *d = dv; const char *s = sv;
    if (((uintptr_t)d & 3) == ((uintptr_t)s & 3)) {
        while (n && ((uintptr_t)d & 3)) { *d++ = *s++; n--; }
        uint32_t *dw = (uint32_t *)d; const uint32_t *sw = (const uint32_t *)s;
        while (n >= 16) { dw[0]=sw[0]; dw[1]=sw[1]; dw[2]=sw[2]; dw[3]=sw[3]; dw+=4; sw+=4; n-=16; }
        while (n >= 4) { *dw++ = *sw++; n -= 4; }
        d = (char *)dw; s = (const char *)sw;
    }
    while (n--) *d++ = *s++;
    return dv;
}
void *memmove(void *dv, const void *sv, size_t n) {
    char *d = dv; const char *s = sv;
    if (d < s) return memcpy(dv, sv, n);
    while (n--) d[n] = s[n];
    return dv;
}
void *memset(void *dv, int c, size_t n) {
    char *d = dv;
    uint32_t w = (uint8_t)c * 0x01010101u;
    while (n && ((uintptr_t)d & 3)) { *d++ = (char)c; n--; }
    uint32_t *dw = (uint32_t *)d;
    while (n >= 16) { dw[0]=w; dw[1]=w; dw[2]=w; dw[3]=w; dw+=4; n-=16; }
    while (n >= 4) { *dw++ = w; n -= 4; }
    d = (char *)dw;
    while (n--) *d++ = (char)c;
    return dv;
}
int memcmp(const void *av, const void *bv, size_t n) {
    const unsigned char *a = av, *b = bv;
    for (; n--; a++, b++)
        if (*a != *b) return *a - *b;
    return 0;
}
size_t strlen(const char *s) { const char *p = s; while (*p) p++; return p - s; }
int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (unsigned char)*a - (unsigned char)*b;
}
int strncmp(const char *a, const char *b, size_t n) {
    for (; n--; a++, b++) {
        if (*a != *b) return (unsigned char)*a - (unsigned char)*b;
        if (!*a) return 0;
    }
    return 0;
}
char *strcpy(char *d, const char *s) { char *r = d; while ((*d++ = *s++)); return r; }
char *strncpy(char *d, const char *s, size_t n) {
    char *r = d;
    for (; n && *s; n--) *d++ = *s++;
    for (; n; n--) *d++ = 0;
    return r;
}
char *strcat(char *d, const char *s) { strcpy(d + strlen(d), s); return d; }
char *strchr(const char *s, int c) {
    for (;; s++) {
        if (*s == (char)c) return (char *)s;
        if (!*s) return 0;
    }
}
char *strrchr(const char *s, int c) {
    const char *r = 0;
    for (;; s++) {
        if (*s == (char)c) r = s;
        if (!*s) return (char *)r;
    }
}
char *strstr(const char *h, const char *n) {
    size_t ln = strlen(n);
    if (!ln) return (char *)h;
    for (; *h; h++)
        if (!strncmp(h, n, ln)) return (char *)h;
    return 0;
}
char *strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}
int strcasecmp(const char *a, const char *b) {
    while (*a && tolower((unsigned char)*a) == tolower((unsigned char)*b)) { a++; b++; }
    return tolower((unsigned char)*a) - tolower((unsigned char)*b);
}
int strncasecmp(const char *a, const char *b, size_t n) {
    for (; n--; a++, b++) {
        int d = tolower((unsigned char)*a) - tolower((unsigned char)*b);
        if (d) return d;
        if (!*a) return 0;
    }
    return 0;
}

/* ---------- math (startup-only paths) ---------- */
double fabs(double x) { return x < 0 ? -x : x; }
double sqrt(double x) {
    if (x <= 0) return 0;
    double g = x > 1 ? x : 1;
    for (int i = 0; i < 40; i++) g = 0.5 * (g + x / g);
    return g;
}
static double norm_angle(double x) {
    const double PI2 = 6.28318530717958647692;
    while (x > 3.14159265358979323846) x -= PI2;
    while (x < -3.14159265358979323846) x += PI2;
    return x;
}
double sin(double x) {
    x = norm_angle(x);
    double t = x, s = x;
    for (int i = 1; i <= 10; i++) {
        t *= -x * x / ((2 * i) * (2 * i + 1));
        s += t;
    }
    return s;
}
double cos(double x) { return sin(x + 1.57079632679489661923); }
double tan(double x) { return sin(x) / cos(x); }
double atan(double x) {
    if (x > 1) return 1.57079632679489661923 - atan(1 / x);
    if (x < -1) return -1.57079632679489661923 - atan(1 / x);
    double t = x, s = x;
    for (int i = 1; i <= 25; i++) {
        t *= -x * x;
        s += t / (2 * i + 1);
    }
    return s;
}
double pow(double x, double y) {
    /* integer exponents only (sufficient here) */
    long n = (long)y;
    double r = 1;
    int neg = n < 0;
    if (neg) n = -n;
    while (n--) r *= x;
    return neg ? 1 / r : r;
}
