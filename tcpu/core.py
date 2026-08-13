"""Constructs the RV32I transformer core: 16 blocks, one forward pass per
instruction.

Token layout (T = 34):
    0        CTL   — PC, pending-load record (flag, rd one-hot, loaded data)
    1        INSTR — instruction word in; all decode/ALU work happens here
    2..33    x0..x31 register tokens (own-index one-hot + 32 value bits)

Microarchitecture by block:
    B0   decode: opcode class / funct3 / rs1 / rs2 / rd one-hots (MLP)
    B1   decode: immediate assembly, control flags, RDW / new-pending fields
    B2   pending-load writeback (attention: reg tokens pull loaded data)
    B3   operand fetch (attention: rs1/rs2/PC routing) + operand muxes
    B4   ADD_Y = BSEL xor SUB; barrel shifter stage 0
    B5   adders generate/propagate (3 adders in parallel); shifter stage 1
    B6-10 Kogge-Stone prefix levels d=1,2,4,8,16; shifter stages 2-4
    B11  sum bits for all 3 adders (A+B, PC+imm, PC+4)
    B12  compare flags EQ / LT / LTU
    B13  RESULT mux; branch-taken
    B14  next-PC mux
    B15  register writeback (attention) + cleanup

Loads take effect via the pending-load path: the core emits addr/size on this
pass, the bus fetches, and B2 of the *next* pass writes the data into rd
(an internalized load-delay slot, invisible architecturally).
"""
from .build import CoreBuilder

T_CTL, T_INSTR, T_REG0 = 0, 1, 2
N_TOKENS = 34

CLASSES = {  # name -> opcode
    "OP": 0x33, "OPIMM": 0x13, "LOAD": 0x03, "STORE": 0x23, "BR": 0x63,
    "LUI": 0x37, "AUIPC": 0x17, "JAL": 0x6F, "JALR": 0x67, "SYS": 0x73,
    "FENCE": 0x0F,
}


def build_core(dtype=None):
    import numpy as np
    if dtype is None:
        dtype = np.float64
    cb = CoreBuilder()
    dm = cb.dims

    # ---- fields ----
    for f in ("TT_CTL", "TT_INSTR", "TT_REG"):
        dm.alloc(f)
    dm.alloc("RIDX", 32)
    dm.alloc("VAL", 32)
    dm.alloc("IW", 32)
    dm.alloc("PC", 32)
    dm.alloc("PLF")
    dm.alloc("PLRD", 32)
    dm.alloc("PLD", 32)
    for c in CLASSES:
        dm.alloc("CLS_" + c)
    dm.alloc("F3", 8)
    dm.alloc("F7B5")
    dm.alloc("RS1", 32)
    dm.alloc("RS2", 32)
    dm.alloc("RD", 32)
    dm.alloc("IMM", 32)
    for f in ("USE_IMM", "SUBF", "LEFT", "RIGHT", "FILL", "NOTJUMP"):
        dm.alloc(f)
    dm.alloc("ARAW", 32)
    dm.alloc("A", 32)
    dm.alloc("BRAW", 32)
    dm.alloc("BSEL", 32)
    dm.alloc("PCLRAW", 32)
    dm.alloc("PCL", 32)
    dm.alloc("ADD_Y", 32)
    dm.alloc("SH", 32)
    for p in ("A", "T", "Q"):  # main, PC+imm target, PC+4
        for f in ("G", "P", "P0", "S"):
            dm.alloc(p + f, 32)
    for f in ("F_EQ", "F_LT", "F_LTU", "BR_TK"):
        dm.alloc(f)
    dm.alloc("RESULT", 32)
    dm.alloc("NPC", 32)
    dm.alloc("RDW", 32)
    dm.alloc("PLF_N")
    dm.alloc("PLRD_N", 32)
    dm.alloc("PWB", 32)
    dm.alloc("WBRAW", 32)

    d = dm.bit
    TI = d("TT_INSTR")
    TR = d("TT_REG")
    blocks = [cb.block() for _ in range(16)]

    def W(b, terms, theta, outs):
        blocks[b].when(TI, terms, theta, outs)

    # ---- B0: primary decode (all on INSTR token) ----
    for name, opc in CLASSES.items():
        v = (opc >> 2) & 0x1F
        terms = []
        npos = 0
        for j in range(5):
            if (v >> j) & 1:
                terms.append((d("IW", j + 2), 1.0))
                npos += 1
            else:
                terms.append((d("IW", j + 2), -1.0))
        W(0, terms, npos, [(d("CLS_" + name), 1.0)])
    for v in range(8):
        terms = []
        npos = 0
        for j in range(3):
            if (v >> j) & 1:
                terms.append((d("IW", j + 12), 1.0))
                npos += 1
            else:
                terms.append((d("IW", j + 12), -1.0))
        W(0, terms, npos, [(d("F3", v), 1.0)])
    W(0, [(d("IW", 30), 1.0)], 1, [(d("F7B5"), 1.0)])
    for field, lo in (("RS1", 15), ("RS2", 20), ("RD", 7)):
        for v in range(32):
            terms = []
            npos = 0
            for j in range(5):
                if (v >> j) & 1:
                    terms.append((d("IW", lo + j), 1.0))
                    npos += 1
                else:
                    terms.append((d("IW", lo + j), -1.0))
            W(0, terms, npos, [(d(field, v), 1.0)])

    # ---- B1: immediates, control flags, writeback / pending pre-compute ----
    def imm_bits(classes, mapping):
        """mapping: imm_bit -> IW bit (or None for const 0)."""
        cterms = [(d("CLS_" + c), 1.0) for c in classes]
        for ib, iwb in mapping.items():
            if iwb is None:
                continue
            W(1, cterms + [(d("IW", iwb), float(len(classes) + 1))],
              len(classes) + 2, [(d("IMM", ib), 1.0)])

    # I-type: OPIMM, LOAD, JALR
    imap = {i: 20 + i for i in range(11)}
    imap.update({i: 31 for i in range(11, 32)})
    imm_bits(("OPIMM", "LOAD", "JALR"), imap)
    # S-type
    smap = {i: 7 + i for i in range(5)}
    smap.update({i: 20 + i for i in range(5, 11)})
    smap.update({i: 31 for i in range(11, 32)})
    imm_bits(("STORE",), smap)
    # B-type
    bmap = {0: None}
    bmap.update({i: 7 + i for i in range(1, 5)})
    bmap.update({i: 20 + i for i in range(5, 11)})
    bmap[11] = 7
    bmap.update({i: 31 for i in range(12, 32)})
    imm_bits(("BR",), bmap)
    # U-type
    umap = {i: None for i in range(12)}
    umap.update({i: i for i in range(12, 32)})
    imm_bits(("LUI", "AUIPC"), umap)
    # J-type
    jmap = {0: None}
    jmap.update({i: 20 + i for i in range(1, 11)})
    jmap[11] = 20
    jmap.update({i: i for i in range(12, 20)})
    jmap.update({i: 31 for i in range(20, 32)})
    imm_bits(("JAL",), jmap)

    for c in ("OPIMM", "LOAD", "STORE", "JALR"):
        W(1, [(d("CLS_" + c), 1.0)], 1, [(d("USE_IMM"), 1.0)])
    W(1, [(d("CLS_OP"), 1.0), (d("F3", 0), 1.0), (d("F7B5"), 1.0)], 3, [(d("SUBF"), 1.0)])
    for c, f3 in (("OP", 2), ("OP", 3), ("OPIMM", 2), ("OPIMM", 3)):
        W(1, [(d("CLS_" + c), 1.0), (d("F3", f3), 1.0)], 2, [(d("SUBF"), 1.0)])
    W(1, [(d("CLS_BR"), 1.0)], 1, [(d("SUBF"), 1.0)])
    W(1, [(d("F3", 1), 2.0), (d("CLS_OP"), 1.0), (d("CLS_OPIMM"), 1.0)], 3, [(d("LEFT"), 1.0)])
    W(1, [(d("F3", 5), 2.0), (d("CLS_OP"), 1.0), (d("CLS_OPIMM"), 1.0)], 3, [(d("RIGHT"), 1.0)])
    W(1, [(d("CLS_JAL"), -1.0), (d("CLS_JALR"), -1.0), (d("CLS_BR"), -1.0)], 0,
      [(d("NOTJUMP"), 1.0)])
    wr_cls = [(d("CLS_" + c), 1.0) for c in ("OP", "OPIMM", "LUI", "AUIPC", "JAL", "JALR")]
    for j in range(1, 32):
        W(1, [(d("RD", j), 2.0)] + wr_cls, 3, [(d("RDW", j), 1.0)])
        W(1, [(d("CLS_LOAD"), 1.0), (d("RD", j), 1.0)], 2, [(d("PLRD_N", j), 1.0)])
    W(1, [(d("CLS_LOAD"), 2.0), (d("RD", 0), -1.0)], 2, [(d("PLF_N"), 1.0)])

    # ---- B2: pending-load writeback ----
    h = blocks[2].head()
    for j in range(32):
        k = [(d("RIDX", j), 1.0)]
        if j > 0:
            k.append((d("PLRD", j), 2.0))
        h.channel([(d("RIDX", j), 1.0)], k)
    for i in range(32):
        h.route(d("PLD", i), d("PWB", i))
        h.route(d("VAL", i), d("PWB", i))
    for i in range(32):
        blocks[2].when(TR, [(d("PWB", i), 2.0)], 1.5, [(d("VAL", i), 1.0)])
        blocks[2].when(TR, [(d("VAL", i), 1.0)], 1, [(d("VAL", i), -1.0)])

    # ---- B3: operand fetch + operand muxes ----
    for field, dst in (("RS1", "ARAW"), ("RS2", "BRAW")):
        h = blocks[3].head()
        for j in range(32):
            h.channel([(d(field, j), 1.0)], [(d("RIDX", j), 1.0)])
        for i in range(32):
            h.route(d("VAL", i), d(dst, i))
    h = blocks[3].head()
    h.channel([(TI, 1.0)], [(d("TT_CTL"), 1.0)])
    for i in range(32):
        h.route(d("PC", i), d("PCLRAW", i))
    for i in range(32):
        W(3, [(d("ARAW", i), 2.0)], 1.5, [(d("A", i), 1.0)])
        W(3, [(d("PCLRAW", i), 2.0)], 1.5, [(d("PCL", i), 1.0)])
        W(3, [(d("USE_IMM"), 1.0), (d("IMM", i), 1.0)], 2, [(d("BSEL", i), 1.0)])
        W(3, [(d("USE_IMM"), -2.0), (d("BRAW", i), 2.0)], 1.5, [(d("BSEL", i), 1.0)])
    W(3, [(d("F7B5"), 2.0), (d("RIGHT"), 2.0), (d("ARAW", 31), 2.0)], 5.5, [(d("FILL"), 1.0)])

    # ---- barrel shifter: stages 0..4 in blocks B4..B8 ----
    for k in range(5):
        b = 4 + k
        sh = 1 << k
        sk = d("BSEL", k)
        src = "A" if k == 0 else "SH"
        for i in range(32):
            out = [(d("SH", i), 1.0)]
            W(b, [(sk, -2.0), (d(src, i), 1.0)], 1, out)
            if i - sh >= 0:
                W(b, [(sk, 1.0), (d("LEFT"), 1.0), (d(src, i - sh), 1.0)], 3, out)
            if i + sh < 32:
                W(b, [(sk, 1.0), (d("RIGHT"), 1.0), (d(src, i + sh), 1.0)], 3, out)
            else:
                W(b, [(sk, 1.0), (d("FILL"), 1.0)], 2, out)
            if k > 0:
                W(b, [(d("SH", i), 1.0)], 1, [(d("SH", i), -1.0)])

    # ---- B4: ADD_Y = BSEL xor SUBF ----
    for i in range(32):
        W(4, [(d("BSEL", i), 1.0), (d("SUBF"), -1.0)], 1, [(d("ADD_Y", i), 1.0)])
        W(4, [(d("SUBF"), 1.0), (d("BSEL", i), -1.0)], 1, [(d("ADD_Y", i), 1.0)])

    # ---- adders: gp in B5, prefix levels B6..B10, sums B11 ----
    def emit_adder(p, X, Y=None, Yconst=None, cin=None):
        G, P, P0, S = p + "G", p + "P", p + "P0", p + "S"
        for i in range(32):
            x = d(X, i)
            if Y is not None:
                y = d(Y, i)
                if i == 0 and cin is not None:
                    W(5, [(x, 1.0), (y, 1.0), (cin, 1.0)], 2, [(d(G, 0), 1.0)])
                else:
                    W(5, [(x, 1.0), (y, 1.0)], 2, [(d(G, i), 1.0)])
                pouts = [(d(P0, i), 1.0), (d(P, i), 1.0)]
                W(5, [(x, 1.0), (y, -1.0)], 1, pouts)
                W(5, [(y, 1.0), (x, -1.0)], 1, pouts)
            else:
                yb = (Yconst >> i) & 1
                pouts = [(d(P0, i), 1.0), (d(P, i), 1.0)]
                if yb:
                    W(5, [(x, 1.0)], 1, [(d(G, i), 1.0)])
                    W(5, [(x, -1.0)], 0, pouts)
                else:
                    W(5, [(x, 1.0)], 1, pouts)
        for lvl, dd in enumerate((1, 2, 4, 8, 16)):
            b = 6 + lvl
            for i in range(dd, 32):
                W(b, [(d(G, i), 2.0), (d(P, i), 1.0), (d(G, i - dd), 1.0)], 2,
                  [(d(G, i), 1.0)])
                W(b, [(d(G, i), 1.0)], 1, [(d(G, i), -1.0)])
                W(b, [(d(P, i), 1.0), (d(P, i - dd), 1.0)], 2, [(d(P, i), 1.0)])
                W(b, [(d(P, i), 1.0)], 1, [(d(P, i), -1.0)])
        for i in range(32):
            if i == 0:
                c = cin
            else:
                c = d(G, i - 1)
            if c is None:
                W(11, [(d(P0, 0), 1.0)], 1, [(d(S, 0), 1.0)])
            else:
                W(11, [(d(P0, i), 1.0), (c, -1.0)], 1, [(d(S, i), 1.0)])
                W(11, [(c, 1.0), (d(P0, i), -1.0)], 1, [(d(S, i), 1.0)])

    emit_adder("A", "A", Y="ADD_Y", cin=d("SUBF"))
    emit_adder("T", "PCL", Y="IMM")
    emit_adder("Q", "PCL", Yconst=4)

    # ---- B12: compare flags ----
    W(12, [(d("AS", i), -1.0) for i in range(32)], 0, [(d("F_EQ"), 1.0)])
    W(12, [(d("AG", 31), -1.0)], 0, [(d("F_LTU"), 1.0)])
    a31, b31, s31 = d("A", 31), d("BSEL", 31), d("AS", 31)
    W(12, [(a31, 1.0), (b31, -1.0)], 1, [(d("F_LT"), 1.0)])
    W(12, [(a31, -1.0), (b31, -1.0), (s31, 1.0)], 1, [(d("F_LT"), 1.0)])
    W(12, [(a31, 1.0), (b31, 1.0), (s31, 1.0)], 3, [(d("F_LT"), 1.0)])

    # ---- B13: RESULT mux + branch-taken ----
    alu2 = [(d("CLS_OP"), 2.0), (d("CLS_OPIMM"), 2.0)]
    for i in range(32):
        out = [(d("RESULT", i), 1.0)]
        W(13, alu2 + [(d("F3", 0), 2.0), (d("AS", i), 1.0)], 5, out)
        W(13, alu2 + [(d("F3", 4), 2.0), (d("A", i), 1.0), (d("BSEL", i), -1.0)], 5, out)
        W(13, alu2 + [(d("F3", 4), 2.0), (d("BSEL", i), 1.0), (d("A", i), -1.0)], 5, out)
        W(13, [(d("CLS_OP"), 3.0), (d("CLS_OPIMM"), 3.0), (d("F3", 6), 3.0),
               (d("A", i), 1.0), (d("BSEL", i), 1.0)], 7, out)
        W(13, alu2 + [(d("F3", 7), 2.0), (d("A", i), 1.0), (d("BSEL", i), 1.0)], 6, out)
        W(13, [(d("LEFT"), 1.0), (d("RIGHT"), 1.0), (d("SH", i), 1.0)], 2, out)
        W(13, [(d("CLS_LUI"), 1.0), (d("IMM", i), 1.0)], 2, out)
        W(13, [(d("CLS_AUIPC"), 1.0), (d("TS", i), 1.0)], 2, out)
        W(13, [(d("CLS_JAL"), 1.0), (d("CLS_JALR"), 1.0), (d("QS", i), 2.0)], 3, out)
    W(13, alu2 + [(d("F3", 2), 2.0), (d("F_LT"), 1.0)], 5, [(d("RESULT", 0), 1.0)])
    W(13, alu2 + [(d("F3", 3), 2.0), (d("F_LTU"), 1.0)], 5, [(d("RESULT", 0), 1.0)])
    br = (d("CLS_BR"), 1.0)
    for f3, flag, pos in ((0, "F_EQ", True), (1, "F_EQ", False), (4, "F_LT", True),
                          (5, "F_LT", False), (6, "F_LTU", True), (7, "F_LTU", False)):
        if pos:
            W(13, [br, (d("F3", f3), 1.0), (d(flag), 1.0)], 3, [(d("BR_TK"), 1.0)])
        else:
            W(13, [(d("CLS_BR"), 1.0), (d("F3", f3), 1.0), (d(flag), -1.0)], 2,
              [(d("BR_TK"), 1.0)])

    # ---- B14: next PC ----
    for i in range(32):
        out = [(d("NPC", i), 1.0)]
        W(14, [(d("BR_TK"), 1.0), (d("TS", i), 1.0)], 2, out)
        W(14, [(d("CLS_JAL"), 1.0), (d("TS", i), 1.0)], 2, out)
        if i > 0:
            W(14, [(d("CLS_JALR"), 1.0), (d("AS", i), 1.0)], 2, out)
        W(14, [(d("NOTJUMP"), 1.0), (d("QS", i), 1.0)], 2, out)
        W(14, [(d("CLS_BR"), 2.0), (d("BR_TK"), -2.0), (d("QS", i), 1.0)], 3, out)

    # ---- B15: register writeback ----
    h = blocks[15].head()
    for j in range(32):
        k = [(d("RIDX", j), 1.0)]
        if j > 0:
            k.append((d("RDW", j), 2.0))
        h.channel([(d("RIDX", j), 1.0)], k)
    for i in range(32):
        h.route(d("RESULT", i), d("WBRAW", i))
        h.route(d("VAL", i), d("WBRAW", i))
    for i in range(32):
        blocks[15].when(TR, [(d("WBRAW", i), 2.0)], 1.5, [(d("VAL", i), 1.0)])
        blocks[15].when(TR, [(d("VAL", i), 1.0)], 1, [(d("VAL", i), -1.0)])

    model = cb.materialize(dtype)
    return model, dm
