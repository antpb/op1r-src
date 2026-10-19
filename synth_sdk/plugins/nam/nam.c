/* nam - Neural Amp Modeler amp, fixed point. Runs one A2 capture at a time, with the same math as NAM core.
 * Profiles are 1.nam .. 16.nam in the nam folder on the disk the OP-1 shows when it is mounted. Leaving disk
 * mode runs a cleanup that deletes every root entry except synth, drum, album, and tape. The effect copies the
 * packs out first and the cleanup then leaves the nam folder where it is. The effect plays the copy. Those
 * files are the fixed-point packs gen_model.py writes, not the JSON captures.
 * The blue encoder's press steps to the next file that is there: the loaded model is dropped and the new one
 * is read into this position's scratch.
 * NAM_BAKED (the host bench and the layer test) takes the same packs from nam_models.h and parses them the same way.
 *   - A2 nano WaveNets (3 channels, 23 dilated layers; a SlimmableContainer's small submodel): wavenet/a2_fast.cpp.
 *     Int16 activations with per-layer power-of-two scales (a2model.py), histories in the scratch buffer, one
 *     Blackfin assembly loop per layer over the block (layer_a2), bit-exact with the C version and a2model.run_int.
 *     While the model's input has been exact zeros for its whole receptive field, every history holds the settled
 *     silence state and the output is a constant: the model is not run (no CPU) until the input moves again.
 *   - LSTMs (up to 2 x 8): lstm.cpp, Q15 (gen_model.py).
 *   knob 0 (blue)  : GAIN   input drive -18 .. +18 dB (0 dB at the middle)
 *   knob 1 (green) : TONE   12 dB/oct low-pass after the amp, 1 kHz .. open (a cab-ish darkening for amp-only captures)
 *   knob 2 (white) : GATE   noise gate open threshold: off at 0, -90 .. -30 dBFS on the input peak (default -50).
 *                           Stays open while the ~6 ms mean is within 18 dB, but not below about -70 dBFS, where the
 *                           hiss mean would hold it. Captures add ~30 dB of small-signal gain: without a gate the
 *                           OP-1 input's hiss becomes static, as with NAM.
 *   blue press     : next profile in the nam folder (1.nam, 2.nam, ...). The turn is still GAIN.
 *   white press    : NAM_BAKED only: page 2 = which baked model (word 7)
 * The listed FX is "nam input" (naminput.c, this folder): it plays the line input once the mic/input key is monitoring
 * it. The built-in mic is left alone, so an open input cannot feed the speaker back through the amp. This file is
 * that FX's code (one copy). FX_nam, which processes
 * the slot's synth instead, is for the host bench and compare tools; it is not registered, so it is not in the list.
 *   knob 3 (red)   : LEVEL  output, square law, unity at 23170, +6 dB at the top
 * Gate, live input (nam input): the open detector is a peak follower (half the step per sample, ~45 ms release) on
 * the signal after a rumble high-pass. A ~6 ms mean sits ~14 dB under a guitar's peaks and barely sees a pick, so a
 * half-volume note never reached -50 dBFS on the average and the gate stayed shut. The peak rests near the hiss
 * (~-59 dBFS; sample spikes ~-58), under the -50 default. It then holds while the mean is within 18 dB, or for
 * ~35 ms after both let go, and fades in ~70 ms (opens in ~1.5 ms). The close level is kept above about -70 dBFS
 * so the hiss mean (~-73) cannot latch it. After it closes, the peak has to fall back under the threshold before
 * another note can open it: a ringing chord and the hiss spikes were re-opening the fast attack. The synth bench
 * keeps the older mean-only gate. Either way the gate scales the model's input, so a closed gate gives the model
 * exact zeros and the output settles to digital silence.
 * Live input only, ahead of the gate: two one-poles at ~48 Hz (low E is about -2.5 dB; 20 Hz is about -16 dB),
 * then a one-pole split at ~8.8 kHz. That top band rests at -12 dB while it is hiss and opens in under a
 * millisecond on a pick, so the steady hiss up there drops about 11 dB and attacks stay bright.
 * Drive is the knob for the whole block. Chasing it per sample dithers the model input, and a capture turns that
 * dither into sputtering once a chord has fallen into the small-signal region and on the noise floor. The head
 * output of a hot capture swings past full scale; the DC blocker takes that offset off and the clip is after it.
 * Output: DC blocker (one pole, ~7 Hz) after the model, then tone and level; loudness normalised to -18 dB from the
 * model's metadata. Changing the model loads its settled state (NAM's prewarm) and fades in over 256 samples; A2
 * histories are filled over ~7 blocks first (silence meanwhile, no SDRAM spike).
 * Models made at 48 kHz run at 44.1 kHz here (no resampler on the OP-1): their response sits ~8 % lower in frequency.
 * Cost: see README (simulator cycle counts: tools/bench.sh).
 * Copyright (C) 2026 antpb. GPL-3.0-or-later with the OP-1 firmware permission (see LICENSE and NOTICE).
 */
#include "op1_plugin.h"

typedef struct {
    uint8_t n_in, q;                    /* inputs (1 or the previous layer's H); W fractional bits */
    const int16_t *W;                   /* [2H][n_in + H][2]: rows i, f, g, o (PyTorch order), interleaved by pairs */
    const int32_t *b;                   /* [4H] at Q(q + 15) */
    const int16_t *h0;                  /* settled state after NAM's 0.5 s prewarm, Q15 */
    const int32_t *c0;
} nam_layer_t;
typedef struct {
    const char *name;
    uint8_t layers, hidden, qh, shape;  /* shape: index into NAM_SHAPES (one specialised block function each) */
    int32_t head_b;                     /* Q(qh + 15) */
    const int16_t *head;                /* [H], qh fractional bits, loudness normalisation folded in */
    nam_layer_t l[2];
    const uint8_t *a2;                  /* A2 nano WaveNet: the pack gen_model.py writes (LSTM fields unused), else 0 */
    uint16_t a2_len;
} nam_model_t;
/* A2 nano WaveNet, 3 channels, 23 layers (a2model.py has the fixed-point formats) */
typedef struct {
    uint8_t K, sz, sr, s_in, sh;        /* kernel; shifts: conv -> z, 1x1 -> r', r into the 1x1 acc, z into head sum */
    uint16_t D, len, off;               /* dilation; history ring entries ((K - 1) D + one block); offset in the ring */
    const int32_t *w;                   /* weight stream, 5K + 15 words in the order layer_a2 reads them (nam_parse) */
    int16_t r0[3];                      /* steady state for silence (NAM's prewarm) */
} nam_a2_layer_t;
typedef struct nam_a2 {
    int16_t head[16 * 4] __attribute__((aligned(4)));   /* [tap][c], c = 3 is 0: one dual-MAC dot per sample */
    int16_t rech[3];                    /* r = sat((rech x) >> s0) */
    uint8_t s0, hsh, ysh;               /* head sum -> head history; head acc -> Q15 output */
    int32_t hb;                         /* head bias (+ rounding), head_scale and loudness folded in */
    int16_t h0[3];                      /* head history steady state */
    int32_t yrest;                      /* model output while every history holds the silence state */
    uint16_t ring, rest;                /* layer ring entries; input samples that settle every history (lookback) */
    nam_a2_layer_t L[23];
} nam_a2_t;
#ifdef NAM_BAKED
#include "nam_models.h"
#else
#define NAM_XS_SHIFT 3
#define NAM_CMAX 262144
#define NAM_MAX_LAYERS 2
#define NAM_MAX_HIDDEN 8
#define NAM_HAS_LSTM 0
#define NAM_NUM_MODELS 1
#define NAM_SHAPES
#endif
#define NAM_A2_RING 8192                /* history entries the scratch holds (A2 nano: 7803) */
#define NAM_A2_BLOCK 64

#define FADE_LEN 256

typedef struct {
    int16_t h[NAM_MAX_LAYERS][NAM_MAX_HIDDEN];   /* entries H.. stay 0: the dot product reads rows padded to pad(H) */
    int32_t c[NAM_MAX_LAYERS][NAM_MAX_HIDDEN];
    int32_t dc;                         /* DC blocker low-pass, Q15 << 10 */
    int32_t lp1, lp2;                   /* tone low-pass states, Q15 */
    uint16_t w[23];                     /* A2: each layer's ring entry for the block's first frame */
    uint32_t quiet;                     /* A2: zero input samples in a row the histories have taken in */
    uint32_t fill;                      /* A2: ring entries set to the steady state so far (spread over blocks) */
    int32_t genv, ghold, ggain;         /* noise gate: mean level (<< 8), open (1) / closed (0), gain (Q15) */
    int32_t gpeak, gholdn;              /* live input: peak level (<< 8), samples left in the hold */
    int32_t hp1, hp2, hflp;             /* live input: rumble high-pass states (Q15 << 10), hiss-band low-pass */
    int32_t hfenv, hfgain;              /* live input: hiss-band level (<< 8), its gain (Q15) */
    uint8_t garm;                       /* gate re-armed: the peak has been back under the open threshold */
    int16_t fade;                       /* samples into the model-change fade-in */
    int8_t  model;                      /* -1 = not loaded */
} nam_state_t;
/* A2 histories live in the FX position's scratch buffer: entries of 4 int16 (3 channels + the input x, which the
 * newest tap's mixin reads). Each ring holds exactly its lookback plus one block, so the whole set is 7803 entries
 * (61 KB) and every block streams through less of the data cache. Filling them takes FILL_CHUNK entries per block
 * (SDRAM: no spike on a model change). */
#define FILL_CHUNK 2048
/* scratch layout: layer rings; a 64-entry ring for the last layer's residual (nothing reads it); one spare word
 * then hs (64 x 3 int32, the head sum); the head history (15 + 64 entries of 4 int16); the pack read from the
 * disk; the parsed model; its weight streams. In the scratch, not on the stack (the audio interrupt's stack is
 * small and shares L1 with voice state). */
#define NAM_A2_SINK_OFF (NAM_A2_RING * 8)
#define NAM_A2_HS_OFF (NAM_A2_SINK_OFF + NAM_A2_BLOCK * 8 + 8)
#define NAM_A2_HL_OFF (NAM_A2_HS_OFF + NAM_A2_BLOCK * 12)
#define NAM_PACK_MAX 8192
#define NAM_A2_PACK_OFF (NAM_A2_HL_OFF + (15 + NAM_A2_BLOCK) * 8)
#define NAM_A2_MODEL_OFF (NAM_A2_PACK_OFF + NAM_PACK_MAX)
#define NAM_A2_WS_OFF (NAM_A2_MODEL_OFF + ((sizeof(nam_a2_t) + 7) & ~7u))
#define NAM_A2_WS_MAX (23 * (5 * 16 + 15))
typedef char nam_ring_check[NAM_A2_WS_OFF + NAM_A2_WS_MAX * 4 <= OP1_FX_SCRATCH_BYTES ? 1 : -1];
static nam_state_t st[3];
static nam_a2_t *nam_a2_of(int pos) { return (nam_a2_t *)((uint8_t *)op1_fx_scratch(pos) + NAM_A2_MODEL_OFF); }
static int32_t *nam_ws_of(int pos) { return (int32_t *)((uint8_t *)op1_fx_scratch(pos) + NAM_A2_WS_OFF); }

#if NAM_HAS_LSTM
/* head length as gen_model.py pads it: even (32-bit pair loads), >= 4 */
static inline int pad(int n) { n += n & 1; return n < 4 ? 4 : n; }

/* all multiplies are int16 x int16 (single-cycle on the Blackfin; int32 x int32 is ~5 cycles) */
static inline int32_t act(const uint32_t *t, int32_t v)         /* v: Q13 over [-8, 8] */
{
    int32_t i = v + 65536;
    i = i < 0 ? 0 : (i > 131071 ? 131071 : i);                  /* MIN / MAX, no branches */
    uint32_t e = t[i >> 8];                                     /* value | delta << 16 */
    return (int16_t)e + (((int16_t)(e >> 16) * (int16_t)(i & 255)) >> 8);
}
#endif

/* sum w[j] x[j], j < n; n even >= 4, both 4-byte aligned. Blackfin: dual MAC, two products and two 32-bit loads
 * per cycle (integer mode, 40-bit accumulators; gen_model.py bounds the sum to int32). */
static inline int32_t dot(const int16_t *w, const int16_t *x, int n)
{
#ifndef __bfin__
    int32_t acc = 0;
    for (int j = 0; j < n; j++) acc += w[j] * x[j];
    return acc;
#else
    int32_t acc;
    asm volatile(
        "I0 = %1;\n\t"
        "I1 = %2;\n\t"
        "A1 = A0 = 0;\n\t"
        "MNOP || R0 = [I0++] || R1 = [I1++];\n\t"
        "LSETUP (1f, 1f) LC0 = %3;\n"
        "1: A1 += R0.H * R1.H, A0 += R0.L * R1.L (IS) || R0 = [I0++] || R1 = [I1++];\n\t"
        "A1 += R0.H * R1.H, A0 += R0.L * R1.L (IS);\n\t"
        "R0 = A0 (IS);\n\t"                                  /* A0 -> even, A1 -> odd register only */
        "R1 = A1 (IS);\n\t"
        "%0 = R0 + R1;"
        : "=d"(acc)
        : "a"(w), "a"(x), "a"((n >> 1) - 1)
        : "R0", "R1", "I0", "I1", "A0", "A1", "LC0", "LT0", "LB0", "memory");
    return acc;
#endif
}

#if NAM_HAS_LSTM
/* a[r] = sum W[r][j] xh[j], r < 2 pairs, j < n (>= 2). W interleaved by row pairs, so one 32-bit load fetches
 * W[2p][j] and W[2p+1][j] and one dual-MAC instruction does both rows: A0 row 2p, A1 row 2p + 1. xh is read as a
 * circular buffer (B1, L1 = n halfwords): it rewinds by itself after every pair. */
static inline void matvec(const int16_t *w, const int16_t *xh, int32_t *a, int pairs, int n)
{
#ifndef __bfin__
    for (int p = 0; p < pairs; p++, w += 2 * n) {
        int32_t s0 = 0, s1 = 0;
        for (int j = 0; j < n; j++) { s0 += w[2 * j] * xh[j]; s1 += w[2 * j + 1] * xh[j]; }
        a[2 * p] = s0;
        a[2 * p + 1] = s1;
    }
#else
    asm volatile(
        "I0 = %1;\n\t"
        "I1 = %2;\n\t"
        "B1 = %2;\n\t"
        "L1 = %5;\n\t"
        "LSETUP (2f, 3f) LC1 = %3;\n"
        "2: A1 = A0 = 0 || R0 = [I0++] || R1.L = W[I1++];\n\t"
        "LSETUP (4f, 4f) LC0 = %4;\n"
        "4: A1 += R0.H * R1.L, A0 += R0.L * R1.L (IS) || R0 = [I0++] || R1.L = W[I1++];\n\t"
        "A1 += R0.H * R1.L, A0 += R0.L * R1.L (IS);\n\t"
        "R0 = A0 (IS);\n\t"                                  /* A0 -> even, A1 -> odd register only */
        "R1 = A1 (IS);\n\t"
        "[%0++] = R0;\n"
        "3: [%0++] = R1;\n\t"
        "L1 = 0;"                                              /* ABI: L registers are 0 outside */
        : "+a"(a)
        : "a"(w), "a"(xh), "a"(pairs), "a"(n - 1), "a"(n * 2)
        : "R0", "R1", "I0", "I1", "B1", "L1", "A0", "A1", "LC0", "LT0", "LB0", "LC1", "LT1", "LB1", "memory");
#endif
}

#endif

static int nam_parse(const uint8_t *p, int n, nam_a2_t *A, int32_t *ws);
#ifdef NAM_BAKED
static const nam_a2_t *baked_a2[3];     /* the parsed A2 model in pos's scratch; 0 for an LSTM or a bad pack */
static void load_model(nam_state_t *s, int m, int pos)
{
    const nam_model_t *M = &NAM_MODELS[m];
    baked_a2[pos] = 0;
    if (M->a2 && nam_parse(M->a2, M->a2_len, nam_a2_of(pos), nam_ws_of(pos))) {
        baked_a2[pos] = nam_a2_of(pos);                      /* rings: filled by fill_a2 over the next blocks */
        s->fill = 0;
    }
    for (int l = 0; l < NAM_MAX_LAYERS; l++)                 /* LSTM state (unused by A2 models: zeros) */
        for (int u = 0; u < NAM_MAX_HIDDEN; u++) {
            int on = !M->a2 && l < M->layers && u < M->hidden;
            s->h[l][u] = on ? M->l[l].h0[u] : 0;
            s->c[l][u] = on ? M->l[l].c0[u] : 0;
        }
    s->model = (int8_t)m;
    s->fade = 0;
}
#endif

#ifndef NAM_BAKED
static void nam_io_reset(int pos);
#endif
static void nam_gate_clear(nam_state_t *s)
{
    s->genv = s->gpeak = s->ghold = s->gholdn = s->ggain = 0;
    s->hp1 = s->hp2 = s->hflp = s->hfenv = 0;
    s->hfgain = 8192;                                        /* hiss-band floor, so the first block is not full-bright */
    s->garm = 1;
}
void nam_init(int pos)
{
    nam_state_t *s = &st[pos % 3];
    s->dc = s->lp1 = s->lp2 = 0;
    nam_gate_clear(s);
    s->model = -1;
#ifndef NAM_BAKED
    nam_io_reset(pos % 3);
#endif
}

#if NAM_HAS_LSTM
/* one LSTM layer: in[ni] -> h (updated in place), c. H and ni are compile-time constants in every caller (run). */
static inline __attribute__((always_inline))
void layer(const nam_layer_t *L, const int H, const int ni, const int16_t *in, int16_t *h, int32_t *c)
{
    int16_t xh[NAM_MAX_HIDDEN * 2];
    int32_t a[4 * NAM_MAX_HIDDEN];
    const int n = ni + H, sh = L->q + 2;                    /* accumulator Q(q + 15) -> Q13 */
#pragma GCC unroll 16
    for (int j = 0; j < ni; j++) xh[j] = in[j];
#pragma GCC unroll 16
    for (int j = 0; j < H; j++) xh[ni + j] = h[j];
    matvec(L->W, xh, a, 2 * H, n);                           /* ifgo = W xh (+ b below) */
    const int32_t *b = L->b;
#pragma GCC unroll 8
    for (int u = 0; u < H; u++) {
        int16_t gi = (int16_t)act(NAM_SIGMOID, (a[u] + b[u]) >> sh);
        int16_t gf = (int16_t)act(NAM_SIGMOID, (a[H + u] + b[H + u]) >> sh);
        int16_t gg = (int16_t)act(NAM_TANH, (a[2 * H + u] + b[2 * H + u]) >> sh);
        int16_t go = (int16_t)act(NAM_SIGMOID, (a[3 * H + u] + b[3 * H + u]) >> sh);
        int32_t cc = c[u];                                   /* Q15, |c| <= 8: f c split into 16 x 16 parts */
        cc = gf * (int16_t)(cc >> 15) + ((gf * (int16_t)(cc & 0x7fff) + 16384) >> 15) + ((gi * gg + 16384) >> 15);
        cc = cc > NAM_CMAX ? NAM_CMAX : (cc < -NAM_CMAX ? -NAM_CMAX : cc);
        c[u] = cc;
        h[u] = (int16_t)((go * (int16_t)act(NAM_TANH, cc >> 2) + 16384) >> 15);
    }
}

#endif

/* a, b: int16 values. gcc emits a 32 x 32 multiply (multi-cycle) when it cannot see that, and a compare + branch
 * pair for each saturation bound. */
static inline int32_t mul16(int32_t a, int32_t b)
{
#ifdef __bfin__
    int32_t r;
    asm("%0 = %1.L * %2.L (IS);" : "=d"(r) : "d"(a), "d"(b));
    return r;
#else
    return (int16_t)a * (int16_t)b;
#endif
}
static inline int32_t sat16(int32_t v)
{
#ifdef __bfin__
    asm("%0 = %0 << 16 (S);\n\t%0 >>>= 16;" : "+d"(v));
    return v;
#else
    return op1_sat16(v);
#endif
}

typedef struct { int16_t g, ta, lvl; } nam_knobs_t;     /* input gain Q15, tone coefficient Q15 (32767 = open), level Q14 */
typedef struct { int32_t dc; int16_t lp1, lp2; } nam_post_t;

/* model output -> DC blocker, tone, level, fade-in. y is the head output and it is allowed to be past full
 * scale: the blocker takes the offset off, then the sample clips. Clipping y first is a square wave. */
static inline __attribute__((always_inline))
int16_t post(nam_state_t *s, nam_post_t *q, int32_t y, nam_knobs_t k)
{
    int32_t d = q->dc >> 10;                                 /* DC blocker, ~7 Hz */
    q->dc += y - d;
    int32_t z = sat16(y - d);
    int32_t ta = k.ta;
    int32_t lp1 = q->lp1 + ((ta * (z - q->lp1)) >> 15);
    int32_t lp2 = q->lp2 + ((ta * (lp1 - q->lp2)) >> 15);
    q->lp1 = (int16_t)lp1;
    q->lp2 = (int16_t)lp2;
    if (ta != 32767) z = (int16_t)lp2;                   /* fully open stays the dry sample; the state still tracks it */
    y = mul16(z, k.lvl) >> 14;
    if (s->fade < FADE_LEN) y = (y * s->fade++) >> 8;
    return (int16_t)sat16(y);
}

#if NAM_HAS_LSTM
/* one block through model M; layers and H are constants (one copy per shape: NAM_SHAPE below) */
static inline __attribute__((always_inline))
void run(nam_state_t *s, const nam_model_t *M, int16_t *buf, nam_knobs_t k, const int layers, const int H)
{
    const int qh = M->qh;
    const int32_t hb = M->head_b;
    nam_post_t q = { s->dc, (int16_t)s->lp1, (int16_t)s->lp2 };
    for (int n = 0; n < OP1_FX_BLOCK; n++) {
        int16_t x = op1_sat16((buf[n] * k.g) >> 15);
        layer(&M->l[0], H, 1, &x, s->h[0], s->c[0]);
        if (layers > 1) layer(&M->l[1], H, H, s->h[0], s->h[1], s->c[1]);
        buf[n] = post(s, &q, (dot(M->head, s->h[layers - 1], pad(H)) + hb) >> qh, k);
    }
    s->dc = q.dc;
    s->lp1 = q.lp1;
    s->lp2 = q.lp2;
}

#endif

/* set ring entries [fill, fill + FILL_CHUNK) to the silence steady state; returns 1 when all are done (then the head
 * history is settled too, every layer starts at ring entry 0 and the model is at rest) */
static int fill_a2(nam_state_t *s, const nam_a2_t *A, int16_t *ring)
{
    uint32_t lo = s->fill, hi = lo + FILL_CHUNK, end = A->ring;
    if (hi > end) hi = end;
    for (int l = 0; l < 23; l++) {
        const nam_a2_layer_t *L = &A->L[l];
        uint32_t i = L->off > lo ? L->off : lo, e = L->off + L->len < hi ? L->off + L->len : hi;
        for (int16_t *p = ring + i * 4; i < e; i++, p += 4) { p[0] = L->r0[0]; p[1] = L->r0[1]; p[2] = L->r0[2]; p[3] = 0; }
    }
    s->fill = hi;
    if (hi != end) return 0;
    int16_t *hl = (int16_t *)((uint8_t *)ring + NAM_A2_HL_OFF);
    for (int j = 0; j < 15; j++) { hl[j * 4] = A->h0[0]; hl[j * 4 + 1] = A->h0[1]; hl[j * 4 + 2] = A->h0[2]; hl[j * 4 + 3] = 0; }
    for (int l = 0; l < 23; l++) s->w[l] = 0;
    s->quiet = A->rest;
    return 1;
}

/* One layer over the block. The previous layer (or run_a2, for layer 0) has already put this block's residual
 * stream entries (r0, r1, r2, x) at ring entries w .. w + 63; entry n is n - w frames into the block, mod len.
 * Per frame f (taps k = 0 .. K-1, tap k is (K-1-k) D frames back; the newest tap is the frame's own entry):
 *   z  = lrelu(sat((cb + sum_k W_k r[f - (K-1-k) D] + mix x) >> sz))      lrelu: max(z, z 328 / 2^15, rounded
 *        to nearest, ties to even: the Blackfin's default convergent rounding of fractional multiplies)
 *   hs[f] += z 2^sh;   next ring entry nw + f = (sat((r 2^s_in + l1 z + lb) >> sr), x)
 * Weight stream (nam_parse), one pass per frame, int32 words, pair(lo, hi) = lo | hi << 16:
 *   cb2, per tap (pair(w[0][2], w[1][2]), pair(w[2][2], 0)), pair(mix2, 0)              output 2
 *   cb0, cb1, per tap and input c pair(w[c][0], w[c][1]), pair(mix0, mix1)              outputs 0, 1
 *   lb0, lb1, pair(2^s_in, 2^s_in), per z input j pair(l1[0][j], l1[1][j])             1x1, outputs 0, 1
 *   lb2, pair(2^s_in, 0), pair(l1[2][0], l1[2][1]), pair(l1[2][2], 0)                    1x1, output 2
 * (a2model.py: formats; a2model.run_int: the same integer math, vectorised.) */
#define NAM_WS_WORDS(K) (5 * (K) + 15)
#ifdef __bfin__
/* Blackfin: one asm frame loop per layer, 5K + 38 instructions a frame. All addressing by circular DAG registers:
 *   I1 (B1/L1 = this ring) walks tap to tap with M1 = D entries - 4 bytes; it reads the newest entry last, steps
 *   back to the frame's oldest tap with M3 for the second pass, and on to the next frame's oldest tap with M0.
 *   I0 (B0/L0 = the weight stream) rewinds by itself: a frame reads the stream exactly once. I2 (B2/L2 = the next
 *   layer's ring) takes the 1x1 outputs; I3 reads hs and P4 writes it back (a frame's last hs word is stored in the
 *   next frame, so P4 starts one word early, on a spare word). Loads run one step ahead (R5: the next cb2).
 *   output 2: 2 cycles per tap (channels 0 + 1 as one dual MAC into A1 + A0, channel 2 single), first
 *   outputs 0 and 1: 3 dual-MAC cycles per tap (weights paired (o0, o1), one input channel per cycle)
 *   mixin: the newest entry's slot 3 holds x (R2.H) times the mixin word.
 *   z: accumulator ASHIFT, saturating extract; lrelu: fractional vector multiply by 328 (rounds: + 2^14) and MAX.
 *   1x1: the biases seed A0/A1, r 2^s_in is one more MAC, then one dual MAC per z; the head sum update shares
 *   instructions with the loads and stores around it. */
typedef struct {
    const int32_t *w; int32_t wlen; int16_t *rg; int32_t len;                     /*  0 */
    const int16_t *start; int32_t m1, m3, m0;                                      /* 16 */
    int16_t *nrg; int32_t nlen; int16_t *nw; int32_t *hs;                          /* 32 */
    int32_t ka, kb, c6, c7;             /* 48: K - 2, K - 1; 328 << 16 | -sz; 2^sh << 16 | -sr */
} nam_layer_args_t;

static void layer_a2(const nam_a2_layer_t *L, int16_t *rg, int w, int16_t *nrg, int nlen, int nw, int32_t *hs)
{
    const int K = L->K, lag = (K - 1) * L->D;
    int st = w - lag;
    if (st < 0) st += L->len;
    nam_layer_args_t a = {
        L->w, NAM_WS_WORDS(K) * 4, rg, L->len * 8,
        rg + st * 4, L->D * 8 - 4, lag * 8, lag * 8 - 8,
        nrg, nlen * 8, nrg + nw * 4, hs,
        K - 2, K - 1, (int32_t)(328u << 16 | (uint32_t)(-L->sz & 0xffff)),
        (int32_t)((1u << L->sh) << 16 | (uint32_t)(-L->sr & 0xffff)) };
    nam_layer_args_t *ap = &a;
    asm volatile(
        "R3 = [%0];\n\t"       "I0 = R3;\n\t" "B0 = R3;\n\t"
        "R3 = [%0 + 4];\n\t"   "L0 = R3;\n\t"
        "R3 = [%0 + 8];\n\t"   "B1 = R3;\n\t"
        "R3 = [%0 + 12];\n\t"  "L1 = R3;\n\t"
        "R3 = [%0 + 16];\n\t"  "I1 = R3;\n\t"
        "R3 = [%0 + 20];\n\t"  "M1 = R3;\n\t"
        "R3 = [%0 + 24];\n\t"  "M3 = R3;\n\t"
        "R3 = [%0 + 28];\n\t"  "M0 = R3;\n\t"
        "R3 = [%0 + 32];\n\t"  "B2 = R3;\n\t"
        "R3 = [%0 + 36];\n\t"  "L2 = R3;\n\t"
        "R3 = [%0 + 40];\n\t"  "I2 = R3;\n\t"
        "R3 = [%0 + 44];\n\t"  "I3 = R3;\n\t"
        "P4 = [%0 + 44];\n\t"
        "P2 = [%0 + 48];\n\t"
        "P3 = [%0 + 52];\n\t"
        "R6 = [%0 + 56];\n\t"
        "R7 = [%0 + 60];\n\t"
        "P4 += -4;\n\t"
        "R5 = [I0++];\n\t"
        "%0 = 64;\n\t"
        "LSETUP (1f, 9f) LC1 = %0;\n"
        /* output 2 */
        "1: A0 = R5 || R0 = [I0++] || R1 = [I1++];\n\t"
        "A1 = R0.H * R1.H, A0 += R0.L * R1.L (IS) || R0 = [I0++] || R2 = [I1 ++ M1];\n\t"
        "A0 += R0.L * R2.L (IS) || R0 = [I0++] || R1 = [I1++];\n\t"
        "LSETUP (2f, 3f) LC0 = P2;\n"
        "2: A1 += R0.H * R1.H, A0 += R0.L * R1.L (IS) || R0 = [I0++] || R2 = [I1 ++ M1];\n"
        "3: A0 += R0.L * R2.L (IS) || R0 = [I0++] || R1 = [I1++];\n\t"
        "A1 += R0.H * R1.H, A0 += R0.L * R1.L (IS) || R0 = [I0++] || R2 = [I1--];\n\t"
        "A0 += R0.L * R2.L (IS) || [P4++] = R3 || R0 = [I0++];\n\t"   /* previous frame's hs[2] */
        "A0 += R0.L * R2.H (IS) || R0 = [I0++] || NOP;\n\t"           /* mixin mix2 x */
        "A0 += A1 || I1 -= M3 || NOP;\n\t"
        "A0 = ASHIFT A0 BY R6.L;\n\t"
        "R3.L = A0 (IS);\n\t"                                          /* z2 */
        /* outputs 0 and 1 */
        "A0 = R0 || R0 = [I0++] || R1 = [I1++];\n\t"
        "A1 = R0 || R0 = [I0++] || NOP;\n\t"
        "LSETUP (4f, 5f) LC0 = P3;\n"
        "4: A1 += R0.H * R1.L, A0 += R0.L * R1.L (IS) || R0 = [I0++] || R2 = [I1 ++ M1];\n\t"
        "A1 += R0.H * R1.H, A0 += R0.L * R1.H (IS) || R0 = [I0++] || NOP;\n"
        "5: A1 += R0.H * R2.L, A0 += R0.L * R2.L (IS) || R0 = [I0++] || R1 = [I1++];\n\t"
        "A1 += R0.H * R1.L, A0 += R0.L * R1.L (IS) || R0 = [I0++] || R2 = [I1--];\n\t"
        "A1 += R0.H * R1.H, A0 += R0.L * R1.H (IS) || R0 = [I0++] || NOP;\n\t"
        "A1 += R0.H * R2.L, A0 += R0.L * R2.L (IS) || R0 = [I0++] || NOP;\n\t"
        "A1 += R0.H * R2.H, A0 += R0.L * R2.H (IS) || R0 = [I0++] || NOP;\n\t"   /* mixin (mix0, mix1) x */
        "A0 = ASHIFT A0 BY R6.L || I1 -= M0 || NOP;\n\t"
        "A1 = ASHIFT A1 BY R6.L;\n\t"
        "R4.H = A1, R4.L = A0 (IS);\n\t"                               /* z1 | z0; R1 = r1 | r0, R2 = x | r2 */
        /* lrelu */
        "R5.H = R4.H * R6.H, R5.L = R4.L * R6.H;\n\t"
        "R4 = MAX (R4, R5) (V);\n\t"
        "R5.L = R3.L * R6.H;\n\t"
        "R3 = MAX (R3, R5) (V);\n\t"
        /* 1x1, outputs 0 and 1 */
        "A0 = R0 || R0 = [I0++] || NOP;\n\t"
        "A1 = R0 || R0 = [I0++] || NOP;\n\t"
        "A1 += R1.H * R0.H, A0 += R1.L * R0.L (IS) || R0 = [I0++] || NOP;\n\t"
        "A1 += R0.H * R4.L, A0 += R0.L * R4.L (IS) || R0 = [I0++] || NOP;\n\t"
        "A1 += R0.H * R4.H, A0 += R0.L * R4.H (IS) || R0 = [I0++] || NOP;\n\t"
        "A1 += R0.H * R3.L, A0 += R0.L * R3.L (IS) || R0 = [I0++] || NOP;\n\t"
        "A0 = ASHIFT A0 BY R7.L || R5 = [I3++] || NOP;\n\t"
        "A1 = ASHIFT A1 BY R7.L;\n\t"
        "R1.H = A1, R1.L = A0 (IS);\n\t"
        /* 1x1, output 2 */
        "A0 = R0 || [I2++] = R1 || R0 = [I0++];\n\t"
        "A0 += R2.L * R0.L (IS) || R0 = [I0++] || NOP;\n\t"
        "A1 = R0.H * R4.H, A0 += R0.L * R4.L (IS) || R0 = [I0++] || NOP;\n\t"
        "A0 += R0.L * R3.L (IS);\n\t"
        "A0 += A1;\n\t"
        "A0 = ASHIFT A0 BY R7.L;\n\t"
        "R2.L = A0 (IS);\n\t"
        /* head sum += z 2^sh; (S) so a hot receptive field clips instead of wrapping */
        "R1 = R4.H * R7.H, R0 = R4.L * R7.H (IS) || [I2++] = R2 || NOP;\n\t"
        "R0 = R0 + R5 (S) || R5 = [I3++] || NOP;\n\t"
        "R1 = R1 + R5 (S) || [P4++] = R0 || R5 = [I3++];\n\t"
        "R0 = R3.L * R7.H (IS) || [P4++] = R1 || NOP;\n"
        "9: R3 = R0 + R5 (S) || R5 = [I0++] || NOP;\n\t"
        "[P4] = R3;\n\t"
        "L0 = 0;\n\t" "L1 = 0;\n\t" "L2 = 0;"                         /* ABI: L registers are 0 outside */
        : "+a"(ap)
        :
        : "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "P2", "P3", "P4", "I0", "I1", "I2", "I3",
          "B0", "B1", "B2", "L0", "L1", "L2", "M0", "M1", "M3", "A0", "A1",
          "LC0", "LT0", "LB0", "LC1", "LT1", "LB1", "CC", "memory");
}
#else
static inline int32_t lo16(int32_t v) { return (int16_t)v; }
static inline int32_t hi16(int32_t v) { return (int16_t)(v >> 16); }
static inline int16_t lrelu(int16_t u)
{
    int32_t v = u * 328, lk = (v + 16384) >> 15;             /* round to nearest, ties to even (as the Blackfin) */
    if ((v & 0x7fff) == 0x4000 && (lk & 1)) lk--;
    return u > lk ? u : (int16_t)lk;
}
static void layer_a2(const nam_a2_layer_t *L, int16_t *rg, int w, int16_t *nrg, int nlen, int nw, int32_t *hs)
{
    const int K = L->K, D = L->D, len = L->len;
    const int32_t sh = 1 << L->sh;
    for (int f = 0; f < NAM_A2_BLOCK; f++) {
        const int16_t *p[16];
        for (int j = 0; j < K; j++) {
            int i = w + f - (K - 1 - j) * D;
            i = i < 0 ? i + len : (i >= len ? i - len : i);
            p[j] = rg + i * 4;
        }
        const int16_t *r = p[K - 1];
        const int32_t *q = L->w;
        int32_t a2 = *q++;
        for (int j = 0; j < K; j++, q += 2) a2 += lo16(q[0]) * p[j][0] + hi16(q[0]) * p[j][1] + lo16(q[1]) * p[j][2];
        a2 += lo16(*q++) * r[3];
        int32_t a0 = *q++, a1 = *q++;
        for (int j = 0; j < K; j++)
            for (int c = 0; c < 3; c++, q++) { a0 += lo16(*q) * p[j][c]; a1 += hi16(*q) * p[j][c]; }
        a0 += lo16(*q) * r[3];
        a1 += hi16(*q++) * r[3];
        int16_t z[3] = { lrelu(op1_sat16(a0 >> L->sz)), lrelu(op1_sat16(a1 >> L->sz)), lrelu(op1_sat16(a2 >> L->sz)) };
        int32_t b0 = *q++, b1 = *q++;
        b0 += lo16(*q) * r[0];
        b1 += hi16(*q++) * r[1];
        for (int j = 0; j < 3; j++, q++) { b0 += lo16(*q) * z[j]; b1 += hi16(*q) * z[j]; }
        int32_t b2 = *q++;
        b2 += lo16(*q++) * r[2];
        b2 += lo16(*q) * z[0] + hi16(*q) * z[1];
        q++;
        b2 += lo16(*q) * z[2];
        int i = nw + f;
        int16_t *o = nrg + (i >= nlen ? i - nlen : i) * 4;
        o[0] = op1_sat16(b0 >> L->sr);
        o[1] = op1_sat16(b1 >> L->sr);
        o[2] = op1_sat16(b2 >> L->sr);
        o[3] = r[3];
        for (int c = 0; c < 3; c++) {
            int64_t acc = (int64_t)hs[f * 3 + c] + (int64_t)z[c] * sh;
            if (acc > 2147483647LL) acc = 2147483647LL;
            else if (acc < -2147483647LL - 1) acc = -2147483647LL - 1;
            hs[f * 3 + c] = (int32_t)acc;
        }
    }
}
#endif

/* one block through an A2 nano WaveNet, layer by layer over the block (same integer math as a2model.run_int) */
static void run_a2(nam_state_t *s, const nam_a2_t *A, int16_t *buf, nam_knobs_t k, int16_t *ring)
{
    int32_t *hs = (int32_t *)((uint8_t *)ring + NAM_A2_HS_OFF);       /* head sum */
    int16_t *hl = (int16_t *)((uint8_t *)ring + NAM_A2_HL_OFF);       /* head history: 15 settled entries, then the block */
    int16_t *sink = (int16_t *)((uint8_t *)ring + NAM_A2_SINK_OFF);
    const nam_a2_layer_t *L = &A->L[0];
    int16_t *e = ring + (L->off + s->w[0]) * 4, *end = ring + (L->off + L->len) * 4;
    const int32_t c0 = A->rech[0], c1 = A->rech[1], c2 = A->rech[2], s0 = A->s0;
    for (int f = 0; f < NAM_A2_BLOCK; f++, e += 4) {
        if (e == end) e = ring + L->off * 4;
        int32_t x = sat16(mul16(buf[f], k.g) >> 15);
        e[0] = (int16_t)sat16(mul16(x, c0) >> s0);
        e[1] = (int16_t)sat16(mul16(x, c1) >> s0);
        e[2] = (int16_t)sat16(mul16(x, c2) >> s0);
        e[3] = (int16_t)x;
        hs[f * 3] = hs[f * 3 + 1] = hs[f * 3 + 2] = 0;
    }
    for (int l = 0; l < 23; l++, L++) {
        if (l < 22) layer_a2(L, ring + L->off * 4, s->w[l], ring + L[1].off * 4, L[1].len, s->w[l + 1], hs);
        else layer_a2(L, ring + L->off * 4, s->w[l], sink, NAM_A2_BLOCK, 0, hs);
        int w = s->w[l] + NAM_A2_BLOCK;
        s->w[l] = (uint16_t)(w >= L->len ? w - L->len : w);
    }
    nam_post_t q = { s->dc, (int16_t)s->lp1, (int16_t)s->lp2 };
    const int32_t hsh = A->hsh, ysh = A->ysh, hb = A->hb;
    for (int f = 0; f < NAM_A2_BLOCK; f++) {
        int16_t *h = hl + (15 + f) * 4;
        h[0] = (int16_t)sat16(hs[f * 3] >> hsh);
        h[1] = (int16_t)sat16(hs[f * 3 + 1] >> hsh);
        h[2] = (int16_t)sat16(hs[f * 3 + 2] >> hsh);
        h[3] = 0;
        buf[f] = post(s, &q, (hb + dot(A->head, hl + f * 4, 64)) >> ysh, k);
    }
    for (int j = 0; j < 15 * 4; j++) hl[j] = hl[NAM_A2_BLOCK * 4 + j];
    s->dc = q.dc;
    s->lp1 = q.lp1;
    s->lp2 = q.lp2;
}

/* the model at rest: its output is the constant yrest, through the same DC blocker, tone, level and fade */
static void rest_a2(nam_state_t *s, const nam_a2_t *A, int16_t *buf, nam_knobs_t k)
{
    nam_post_t q = { s->dc, (int16_t)s->lp1, (int16_t)s->lp2 };
    for (int f = 0; f < NAM_A2_BLOCK; f++) buf[f] = post(s, &q, A->yrest, k);
    s->dc = q.dc;
    s->lp1 = q.lp1;
    s->lp2 = q.lp2;
}

#if NAM_HAS_LSTM
#define NAM_SHAPE(L, H) \
    static void run_##L##x##H(nam_state_t *s, const nam_model_t *M, int16_t *buf, nam_knobs_t k) { run(s, M, buf, k, L, H); }
NAM_SHAPES
#undef NAM_SHAPE
#define NAM_SHAPE(L, H) run_##L##x##H,
static void (*const RUN[])(nam_state_t *, const nam_model_t *, int16_t *, nam_knobs_t) = { NAM_SHAPES };
#undef NAM_SHAPE
#endif

/* 2^(e / 4096) for e in [-6 * 4096, 0], Q15 (per block) */
/* ---- live audio input ----
 * The audio callback is called as (out block, in block, ...): the firmware's test modes show it (0x0191A7A8 copies
 * in -> out, 0x0191A880 writes tones to out = R0). op1r's callback wrapper (screens/sysmon.c) copies the in block to
 * op1_audio_in[] at the start of each callback when op1_audio_in_on is set; this FX runs inside the same callback, so
 * it reads exactly the current block: 64 frames of (L, R) 32-bit words, MSB aligned. (A first version followed the
 * RX DMA position itself and read inconsistent blocks.) */
/* T1 resource monitor line (screens/sysmon.c): source block address, largest raw word, OR of low 16 bits, blocks */
volatile uint32_t nam_diag[4];

/* The mic/input key writes 2 here while external audio is monitored, and 0 when it is turned off
 * (global_controller on_key, code 0x2B). The word beside it is the source: 0 is the built-in mic,
 * 1 is the line jack (plugging a cable switches 0 -> 1, unplugging switches 1 -> 0). */
#ifndef OP1_HOST
#define NAM_IN_MON   (*(volatile int32_t *)0xFF8010B0)
#define NAM_IN_SRC   (*(volatile int32_t *)0xFF8010B4)
static int nam_line_live(void) { return NAM_IN_MON == 2 && NAM_IN_SRC == 1; }
#endif

/* channel: whichever of L / R is louder (block energy, smoothed; switches at 2x), so a mono cable works on
 * either side */
#ifndef OP1_HOST
static int32_t in_eL, in_eR;
static uint8_t in_right;
#endif
static void read_input(int16_t *in)
{
#ifdef OP1_HOST
    for (int n = 0; n < OP1_FX_BLOCK; n++) in[n] = 0;
#else
    if (!op1_audio_in_on) {                                  /* first call: the copy starts with the next callback */
        op1_audio_in_on = 1;
        for (int n = 0; n < OP1_FX_BLOCK; n++) in[n] = 0;
        return;
    }
    uint32_t big = nam_diag[1], lo = 0;
    int32_t ab = (int32_t)big < 0 ? -(int32_t)big : (int32_t)big, eL = 0, eR = 0;
    for (int n = 0; n < OP1_FX_BLOCK; n++) {
        int32_t L = op1_audio_in[2 * n], R = op1_audio_in[2 * n + 1];
        int32_t aL = L < 0 ? -L : L, aR = R < 0 ? -R : R;
        if (aL > ab) { big = (uint32_t)L; ab = aL; }
        if (aR > ab) { big = (uint32_t)R; ab = aR; }
        lo |= (uint32_t)(L | R) & 0xffff;
        eL += aL >> 16; eR += aR >> 16;
    }
    in_eL += (eL - in_eL) >> 3; in_eR += (eR - in_eR) >> 3;
    /* >> 22 was deaf around -54 dBFS, so a quiet guitar on the ring never won. +256 is the old +4. */
    if (!in_right && in_eR > 2 * in_eL + 256) in_right = 1;
    else if (in_right && in_eL > 2 * in_eR + 256) in_right = 0;
    for (int n = 0; n < OP1_FX_BLOCK; n++) in[n] = (int16_t)(op1_audio_in[2 * n + in_right] >> 16);
    nam_diag[0] = (uint32_t)op1_audio_in_src;
    nam_diag[1] = big;
    nam_diag[2] |= lo;
    nam_diag[3]++;
#endif
}

static int32_t pow2_q15(int32_t e)
{
    int32_t k = (-e + 4095) >> 12;                           /* whole octaves down, e + 4096 k in [0, 4096) */
    int32_t f = e + (k << 12);                               /* Q12 fraction */
    int32_t m = 32768 + ((f * (22795 + ((f * 9973) >> 12))) >> 12);   /* 2^f ~ 1 + f (0.6957 + 0.3043 f) */
    return m >> k;
}

static int rd_u16(const uint8_t *p) { return p[0] | (p[1] << 8); }
static int rd_i16(const uint8_t *p) { return (int16_t)rd_u16(p); }
static int32_t rd_i32(const uint8_t *p)
{
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}
static int32_t pair16(int lo, int hi) { return (int32_t)((uint32_t)(lo & 0xffff) | (uint32_t)hi << 16); }
/* A2 pack (gen_model.py pack_a2: little endian, "NA2" 1) -> A, and the weight streams layer_a2 reads -> ws. The
 * pack's ring sizes and offsets (powers of two, which older loaders require) are not used: here each ring is exactly
 * its lookback plus one block. */
static int nam_parse(const uint8_t *p, int n, nam_a2_t *A, int32_t *ws)
{
    if (n < 32 || p[0] != 'N' || p[1] != 'A' || p[2] != '2' || p[3] != 1) return 0;
    int i = 6;
    if (rd_u16(p + i) != 23) return 0;
    i += 2;
    for (int c = 0; c < 3; c++) { A->rech[c] = (int16_t)rd_i16(p + i); i += 2; }
    A->s0 = p[i]; A->hsh = p[i + 1]; A->ysh = p[i + 2]; i += 4;
    if (A->s0 == 0 || A->s0 > 24 || A->hsh > 24 || A->ysh > 24) return 0;
    A->hb = rd_i32(p + i); i += 4;
    for (int c = 0; c < 3; c++) { A->h0[c] = (int16_t)rd_i16(p + i); i += 2; }
    i += 2;
    i = (i + 3) & ~3;
    if (i + 96 > n) return 0;
    for (int j = 0; j < 16; j++) {
        for (int c = 0; c < 3; c++) A->head[j * 4 + c] = (int16_t)rd_i16(p + i + (j * 3 + c) * 2);
        A->head[j * 4 + 3] = 0;
    }
    i += 96;
    uint32_t ring = 0, rest = 15;
    int32_t *w = ws;
    for (int l = 0; l < 23; l++) {
        i = (i + 3) & ~3;
        if (i + 66 > n) return 0;
        nam_a2_layer_t *L = &A->L[l];
        L->K = p[i]; L->sz = p[i + 1]; L->sr = p[i + 2]; L->s_in = p[i + 3]; L->sh = p[i + 4];
        i += 6;
        L->D = (uint16_t)rd_u16(p + i);
        i += 6;
        const int K = L->K;
        if (K < 3 || K > 16 || L->D < 1 || L->sz > 24 || L->sr > 24 || L->s_in > 14 || L->sh > 14) return 0;
        uint32_t lag = (uint32_t)(K - 1) * L->D;
        L->len = (uint16_t)(lag + NAM_A2_BLOCK);
        L->off = (uint16_t)ring;
        ring += lag + NAM_A2_BLOCK;
        rest += lag;
        if (ring > NAM_A2_RING) return 0;
        int32_t cb[3], lb[3];
        int16_t l1[9];
        for (int c = 0; c < 3; c++) { cb[c] = rd_i32(p + i); i += 4; }
        for (int c = 0; c < 9; c++) { l1[c] = (int16_t)rd_i16(p + i); i += 2; }
        i += 2;
        for (int c = 0; c < 3; c++) { lb[c] = rd_i32(p + i); i += 4; }
        for (int c = 0; c < 3; c++) { L->r0[c] = (int16_t)rd_i16(p + i); i += 2; }
        int nw01 = rd_u16(p + i); i += 2;
        int nw2 = rd_u16(p + i); i += 2;
        if (nw01 != 6 * K + 8 || nw2 != 4 * K + 6) return 0;
        i = (i + 3) & ~3;
        if (i + (nw01 + nw2) * 2 > n) return 0;
        const uint8_t *w01 = p + i, *w2 = p + i + nw01 * 2;   /* gen_model.py a2_layer_weights */
        i += (nw01 + nw2) * 2;
        L->w = w;
        *w++ = cb[2];
        for (int k = 0; k < K; k++) {
            *w++ = pair16(rd_i16(w2 + 8 * k), rd_i16(w2 + 8 * k + 2));
            *w++ = pair16(rd_i16(w2 + 8 * k + 4), 0);
        }
        *w++ = pair16(rd_i16(w2 + 8 * K), 0);
        *w++ = cb[0];
        *w++ = cb[1];
        for (int k = 0; k < 3 * K + 1; k++) *w++ = pair16(rd_i16(w01 + 4 * k), rd_i16(w01 + 4 * k + 2));
        *w++ = lb[0];
        *w++ = lb[1];
        *w++ = pair16(1 << L->s_in, 1 << L->s_in);
        for (int j = 0; j < 3; j++) *w++ = pair16(l1[j], l1[3 + j]);
        *w++ = lb[2];
        *w++ = pair16(1 << L->s_in, 0);
        *w++ = pair16(l1[6], l1[7]);
        *w++ = pair16(l1[8], 0);
    }
    A->ring = (uint16_t)ring;
    A->rest = (uint16_t)rest;
    int32_t y = A->hb;
    for (int j = 0; j < 16; j++)
        for (int c = 0; c < 3; c++) y += A->head[j * 4 + c] * A->h0[c];
    A->yrest = y >> A->ysh;
    return 1;
}

#ifndef NAM_BAKED
/* Profiles are 1.nam .. 16.nam in the nam folder of the mounted disk (fixed-point packs from gen_model.py).
 * One pack is read into this position's scratch, after the rings. Blue press steps op1_fx_step and the next
 * file replaces it. A missing number is skipped. A file that is not a pack is kept as the selection and marked bad. */
#ifdef OP1_HOST
#include <stdio.h>
#ifndef NAM_PACK_DIR
#define NAM_PACK_DIR "plugins/nam/packs"
#endif
volatile uint8_t op1_fx_step[3][4];
static FILE *nam_fp[3];
#endif
#define NAM_NFILES 16
enum { NAM_IDLE = 0, NAM_READ = 1, NAM_READY = 2, NAM_BAD = 3, NAM_NONE = 4 };
typedef struct {
    int fd, got, num, loaded;
    uint32_t mask;
    uint8_t phase, step, reset, wait;
} nam_io_t;
static nam_io_t nam_io[3];
static uint8_t nam_io_live[3];
static volatile int8_t nam_ui_num[3], nam_ui_st[3];

static uint8_t *nam_pack_of(int pos) { return (uint8_t *)op1_fx_scratch(pos) + NAM_A2_PACK_OFF; }

static void nam_path(char *o, int n)
{
#ifdef OP1_HOST
    const char *s = NAM_PACK_DIR "/";
#else
    const char *s = "/yaffs2/user/nam/";
#endif
    while (*s) *o++ = *s++;
    if (n >= 10) *o++ = (char)('0' + n / 10);
    *o++ = (char)('0' + (n % 10));
    *o++ = '.'; *o++ = 'n'; *o++ = 'a'; *o++ = 'm';
    *o = 0;
}
static int nam_exists(const char *path)
{
#ifdef OP1_HOST
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
#else
    return ((int (*)(const char *, int))0x01A9A6B0)(path, 0) == 0;
#endif
}
static int nam_open_pos(int pos, const char *path)
{
#ifdef OP1_HOST
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    nam_fp[pos] = f;
    return 1;
#else
    (void)pos;
    return ((int (*)(const char *, int, int))0x01A997F4)(path, 0, 0x100);
#endif
}
static int nam_read_pos(int pos, void *b, int n)
{
#ifdef OP1_HOST
    if (!nam_fp[pos]) return -1;
    return (int)fread(b, 1, (size_t)n, nam_fp[pos]);
#else
    return ((int (*)(int, void *, unsigned))0x01A99C90)(nam_io[pos].fd, b, (unsigned)n);
#endif
}
static void nam_close_pos(int pos)
{
#ifdef OP1_HOST
    if (nam_fp[pos]) { fclose(nam_fp[pos]); nam_fp[pos] = 0; }
#else
    if (nam_io[pos].fd >= 0) ((int (*)(int))0x01A99908)(nam_io[pos].fd);
#endif
    nam_io[pos].fd = -1;
}
static void nam_publish(int pos)
{
    const nam_io_t *io = &nam_io[pos];
    int st = 1;
    if (io->phase == NAM_READY) st = 2;
    else if (io->phase == NAM_BAD) st = 3;
    else if (io->phase == NAM_NONE || io->phase == NAM_IDLE) st = 0;
    nam_ui_num[pos] = (int8_t)io->num;
    nam_ui_st[pos] = (int8_t)st;
}
static void nam_fail(int pos)
{
    nam_close_pos(pos);
    nam_io[pos].phase = NAM_BAD;
    nam_io[pos].loaded = 0;
}
static void nam_read_some(int pos)
{
    nam_io_t *io = &nam_io[pos];
    int room = NAM_PACK_MAX - io->got;
    int ask = room;
    if (ask <= 0) { nam_fail(pos); return; }
    int n = nam_read_pos(pos, nam_pack_of(pos) + io->got, ask);
    if (n < 0) { nam_fail(pos); return; }
    io->got += n;
    if (n == ask && io->got < NAM_PACK_MAX) return;
    if (n == ask && io->got == NAM_PACK_MAX) {
        uint8_t extra;
        int k = nam_read_pos(pos, &extra, 1);
        if (k > 0) { nam_fail(pos); return; }
    }
    nam_close_pos(pos);
    if (io->got < 32 || !nam_parse(nam_pack_of(pos), io->got, nam_a2_of(pos), nam_ws_of(pos))) { nam_fail(pos); return; }
    io->phase = NAM_READY;
    io->loaded = io->num;
}
static void nam_service(int pos)
{
    nam_io_t *io = &nam_io[pos];
    if (io->reset) {
        nam_close_pos(pos);
        io->reset = 0;
        io->fd = -1;
        io->got = 0;
        io->num = 0;
        io->loaded = 0;
        io->phase = NAM_IDLE;
    }
    uint8_t step = op1_fx_step[pos][0];
    int clicked = step != io->step;
    if (clicked) io->step = step;
    if (!clicked && io->phase == NAM_NONE && io->wait != 0xff) {
        io->wait++;
        if ((io->wait & 31) == 0) io->phase = NAM_IDLE;
    }
    if (!clicked && io->phase == NAM_READ) { nam_read_some(pos); nam_publish(pos); return; }
    if (!clicked && io->phase != NAM_IDLE) { nam_publish(pos); return; }
    if (io->fd >= 0 || io->phase == NAM_READ) nam_close_pos(pos);
    uint32_t mask = 0;
    for (int n = 1; n <= NAM_NFILES; n++) {
        char path[40];
        nam_path(path, n);
        if (nam_exists(path)) mask |= 1u << n;
    }
    io->mask = mask;
    int start = (clicked && io->num) ? (io->num % NAM_NFILES) + 1 : 1;
    int found = 0;
    for (int k = 0; k < NAM_NFILES; k++) {
        int c = (start - 1 + k) % NAM_NFILES + 1;
        if (mask & (1u << c)) { found = c; break; }
    }
    if (!found) {
        io->phase = NAM_NONE;
        io->num = 0;
        io->loaded = 0;
        io->wait = 0;
        nam_publish(pos);
        return;
    }
    if (found == io->loaded) {
        io->num = found;
        io->phase = NAM_READY;
        nam_publish(pos);
        return;
    }
    char path[40];
    nam_path(path, found);
    int fd = nam_open_pos(pos, path);
    io->num = found;
    io->loaded = 0;
    io->got = 0;
    if (fd < 0) { io->phase = NAM_BAD; nam_publish(pos); return; }
    io->fd = fd;
    io->phase = NAM_READ;
    nam_read_some(pos);
    nam_publish(pos);
}
static void nam_io_reset(int pos)
{
#ifdef OP1_HOST
    if (nam_io_live[pos]) nam_close_pos(pos);
    nam_io[pos].fd = -1;
#else
    if (!nam_io_live[pos]) nam_io[pos].fd = -1;
#endif
    nam_io_live[pos] = 1;
    nam_io_t *io = &nam_io[pos];
    io->got = 0;
    io->num = 0;
    io->loaded = 0;
    io->mask = 0;
    io->phase = NAM_IDLE;
    io->reset = 1;
    io->wait = 0;
    io->step = op1_fx_step[pos][0];
    nam_ui_num[pos] = 0;
    nam_ui_st[pos] = 0;
}
/* The audio interrupt must not touch the disk. ci_service (the main loop) polls the copy made when disk
 * mode ends. nam_del_gate runs at the start of the cleanup that would otherwise delete the nam folder, while
 * the volume is still mounted. The file object lives on this stack frame: a static one at 0x01EED500 faulted
 * in f_open on the boot screen (SEQSTAT 0x24, RETX 0x199B2C8). The stock opener uses a 0x844 frame. */
#ifndef OP1_HOST
static void nam_put_nam(char *o, int n)
{
    if (n >= 10) *o++ = (char)('0' + n / 10);
    *o++ = (char)('0' + (n % 10));
    *o++ = '.'; *o++ = 'n'; *o++ = 'a'; *o++ = 'm';
    *o = 0;
}
static int nam_import_profiles(void)
{
    uint32_t fil[0x211];
    uint8_t buf[512];
    int made = 0;
    for (int n = 1; n <= NAM_NFILES; n++) {
        char fat[16], yaf[28];
        const char *s = "/nam/";
        char *o = fat;
        while (*s) *o++ = *s++;
        nam_put_nam(o, n);
        s = "/yaffs2/user/nam/";
        o = yaf;
        while (*s) *o++ = *s++;
        nam_put_nam(o, n);
        int fr = ((int (*)(void *, const char *, unsigned))0x199B2A8)(fil, fat, 1);
        if (fr == 12) return 0;
        if (fr) {
            if (fr == 4 || fr == 5) ((int (*)(const char *))0x01A9A3FC)(yaf);
            continue;
        }
        if (!made) {
            ((int (*)(const char *, int))0x01A9A888)("/yaffs2/user/nam", 0x1FF);
            made = 1;
        }
        int out = ((int (*)(const char *, int, int))0x01A997F4)(yaf, 0x241, 0x180);
        if (out < 0) {
            ((int (*)(void *))0x199BD7C)(fil);
            continue;
        }
        int bad = 0;
        for (;;) {
            unsigned got = 0;
            if (((int (*)(void *, void *, unsigned, unsigned *))0x199B5C4)(fil, buf, 512, &got) || !got)
                break;
            int w = ((int (*)(int, const void *, unsigned))0x01A99F84)(out, buf, got);
            if (w < 0 || (unsigned)w != got) { bad = 1; break; }
        }
        ((int (*)(int))0x01A99908)(out);
        ((int (*)(void *))0x199BD7C)(fil);
        if (bad) ((int (*)(const char *))0x01A9A3FC)(yaf);
    }
    return 1;
}
static int nam_last_is_nam(const char *p)
{
    const char *b = p;
    int i = 0;
    if (!p) return 0;
    for (; i < 64 && p[i]; i++) if (p[i] == '/') b = p + i + 1;
    if (i == 64) return 0;
    return b[0] == 'n' && b[1] == 'a' && b[2] == 'm' && b[3] == 0;
}
int nam_del_skip(const char *path)
{
    static uint8_t done;
    if (!done && nam_import_profiles()) done = 1;
    return nam_last_is_nam(path);
}
#endif
void nam_disk_poll(void)
{
#ifndef OP1_HOST
    for (int pos = 0; pos < 3; pos++)
        if (nam_io_live[pos]) nam_service(pos);
#endif
}
static int nam_acquire(nam_state_t *s, int pos)
{
    if (nam_io[pos].phase != NAM_READY) { s->model = -1; return 0; }
    if (s->model != nam_io[pos].loaded) {
        s->fill = 0;
        s->fade = 0;
        s->model = (int8_t)nam_io[pos].loaded;
    }
    return 1;
}
int nam_file_num(int pos) { return nam_ui_num[pos % 3]; }
int nam_file_state(int pos) { return nam_ui_st[pos % 3]; }
#endif

/* Live-input conditioner. Rumble high-pass (two one-poles, coef 223 ~ 48 Hz, state Q15<<10), then the gate,
 * then the hiss band. The peak follower is what lets a half-volume note open the gate: the mean of that note
 * sits ~14 dB under the peak, which is why the old mean detector stayed shut. Release >> 11 is ~45 ms, so the
 * envelope does not ripple at the fundamental. Close is 18 dB under the peak threshold because the mean is
 * already ~14 dB down; NAM_CLOSE_FLOOR keeps that from sinking into the hiss mean and latching. */
#define NAM_RUMBLE 223
#define NAM_HF_LP 23415
#define NAM_HF_FLOOR 8192
#define NAM_GATE_HOLD 1540
#define NAM_CLOSE_FLOOR (10 << 8)

static int32_t nam_hpole(int32_t x, int32_t *s)
{
    int32_t d = *s >> 10;
    *s += ((x - d) * NAM_RUMBLE) >> 5;
    return x - (*s >> 10);
}

static void nam_condition_input(nam_state_t *s, int16_t *buf, int32_t kw)
{
    int32_t hp1 = s->hp1, hp2 = s->hp2, hflp = s->hflp;
    int32_t peak = s->gpeak, mean = s->genv, on = s->ghold, hold = s->gholdn, gg = s->ggain;
    int32_t hfenv = s->hfenv, hfg = s->hfgain;
    int armed = s->garm;
    int32_t open = 0, close = 0, hinv = 0;
    if (kw) {
        open = pow2_q15((kw * 5 >> 2) - 15 * 4096) << 8;    /* 2^-15 .. 2^-5: -90 .. -30 dBFS */
        close = open >> 3;
        if (close < NAM_CLOSE_FLOOR) close = NAM_CLOSE_FLOOR;
        int32_t ho = open >> 9;                              /* (open/2) in Q15, the hiss band's full-open level */
        if (ho < 1) ho = 1;
        hinv = (int32_t)(((int64_t)32767 << 8) / ho);
    }
    for (int n = 0; n < OP1_FX_BLOCK; n++) {
        int32_t x = op1_sat16(nam_hpole(nam_hpole(buf[n], &hp1), &hp2));
        int32_t a8 = (x < 0 ? -x : x) << 8;
        if (a8 > peak) peak += (a8 - peak) >> 1;
        else peak += (a8 - peak) >> 11;
        mean += (a8 - mean) >> 8;
        hflp += ((x - hflp) * NAM_HF_LP) >> 15;
        int32_t high = x - hflp;
        int32_t ha8 = (high < 0 ? -high : high) << 8;
        if (ha8 > hfenv) hfenv += (ha8 - hfenv) >> 2;
        else hfenv += (ha8 - hfenv) >> 10;
        if (!kw) {
            buf[n] = (int16_t)x;
            continue;
        }
        if (peak < open) armed = 1;
        if (!on) {
            if (armed && peak >= open) { on = 1; hold = NAM_GATE_HOLD; armed = 0; }
        } else if (mean >= close) {
            hold = NAM_GATE_HOLD;
        } else if (hold > 0) {
            hold--;
        } else {
            on = 0;
            armed = 0;
        }
        gg += on ? ((32767 - gg) * 496) >> 15 : -((gg * 11 + 32767) >> 15);
        if (gg < 0) gg = 0;
        int32_t lin = 32767;
        if (hfenv < (open >> 1)) {
            lin = ((hfenv >> 8) * hinv) >> 8;
            if (lin > 32767) lin = 32767;
            if (lin < 0) lin = 0;
        }
        int32_t sq = (lin * lin) >> 15;
        int32_t ht = NAM_HF_FLOOR + ((sq * (32767 - NAM_HF_FLOOR)) >> 15);
        if (ht > hfg) hfg += (ht - hfg) >> 3;
        else hfg += (ht - hfg) >> 8;
        int32_t y = hflp + ((high * (hfg >> 3)) >> 12);
        buf[n] = op1_sat16((op1_sat16(y) * gg) >> 15);
    }
    if (!kw) { on = 0; hold = 0; gg = 0; armed = 1; }
    s->hp1 = hp1; s->hp2 = hp2; s->hflp = hflp;
    s->gpeak = peak; s->genv = mean; s->ghold = on; s->gholdn = hold; s->ggain = gg;
    s->hfenv = hfenv; s->hfgain = hfg;
    s->garm = (uint8_t)armed;
}

/* both FX: input = 0 processes buf (the synth), 1 replaces it with the live input first (naminput.c).
 * Returns 1 when the model ran. */
static int nam_block(int pos, int16_t *buf, const int16_t *p, int input)
{
    nam_state_t *s = &st[pos % 3];
#if defined(OP1_HOST) && !defined(NAM_BAKED)
    nam_service(pos % 3);
#endif
#ifndef OP1_HOST
    if (input && !nam_line_live()) {                         /* mic, or the mic key's monitor still off */
        for (int n = 0; n < OP1_FX_BLOCK; n++) buf[n] = 0;
        nam_gate_clear(s);
        s->model = -1;                                       /* next arm reloads the settled silence state */
        return 0;
    }
#endif
    int32_t kg = op1_fx_param_q15(p, 0), kt = op1_fx_param_q15(p, 1);
    int32_t kw = op1_fx_param_q15(p, 2), kl = op1_fx_param_q15(p, 3);
    const nam_a2_t *A;
#ifdef NAM_BAKED
    int32_t km = p[7] < 0 ? 0 : p[7];                        /* white page 2: model */
    int m = NAM_NUM_MODELS > 1 ? (km * NAM_NUM_MODELS) >> 15 : 0;
    if (m >= NAM_NUM_MODELS) m = NAM_NUM_MODELS - 1;
    if (m != s->model) load_model(s, m, pos);
    A = baked_a2[pos];
#else
    if (!nam_acquire(s, pos % 3)) {
        for (int n = 0; n < OP1_FX_BLOCK; n++) buf[n] = 0;
        return 0;
    }
    A = nam_a2_of(pos % 3);
#endif
    nam_knobs_t k;
    /* gain: -18 .. +18 dB = 2^-3 .. 2^3, over the model's input headroom 2^NAM_XS_SHIFT: Q15 2^-6 .. 2^0 */
    int32_t g = pow2_q15(((kg * 6) >> 3) - 6 * 4096);       /* kg * 6 * 4096 / 32768 */
    k.g = (int16_t)(g > 32767 ? 32767 : g);
    /* tone: one-pole coefficient 1 - exp(-2 pi fc / fs), fc = 1 kHz * 2^(4 x) up to 16 kHz ~ 0.13 .. 0.9; open at top */
    k.ta = (int16_t)(kt >= 32000 ? 32767 : 4300 + ((kt * kt >> 15) * 25200 >> 15));
    k.lvl = (int16_t)((kl * kl) >> 15);                      /* Q14: unity at 16384 */
    if (input) read_input(buf);
    if (input) nam_condition_input(s, buf, kw);
    else if (kw) {                                           /* synth bench: mean detector, 12 dB hysteresis */
        int32_t open = pow2_q15((kw * 5 >> 2) - 15 * 4096) << 8;
        int32_t close = open >> 2;
        int32_t env = s->genv, on = s->ghold, gg = s->ggain;
        for (int n = 0; n < OP1_FX_BLOCK; n++) {
            int32_t a = buf[n] < 0 ? -buf[n] : buf[n];
            env += ((a << 8) - env) >> 8;
            if (on) { if (env < close) on = 0; }
            else if (env >= open) on = 1;
            gg += on ? ((32767 - gg) * 496) >> 15 : -((gg * 11 + 32767) >> 15);
            if (gg < 0) gg = 0;
            buf[n] = (int16_t)((buf[n] * gg) >> 15);
        }
        s->genv = env; s->ghold = on; s->ggain = gg;
    }
    if (A) {
        int16_t *ring = (int16_t *)op1_fx_scratch(pos);
        if (s->fill < A->ring && !fill_a2(s, A, ring)) {
            for (int n = 0; n < OP1_FX_BLOCK; n++) buf[n] = 0;     /* rings still filling: ~1.5 ms per chunk */
            return 0;
        }
        int16_t any = 0;
        for (int n = 0; n < OP1_FX_BLOCK; n++) any |= buf[n];
#ifndef NAM_NO_REST
        if (!any && s->quiet >= A->rest) {                   /* every history settled: nothing to compute */
            rest_a2(s, A, buf, k);
            return 0;
        }
#endif
        run_a2(s, A, buf, k, ring);
        s->quiet = any ? 0 : s->quiet + OP1_FX_BLOCK;
        return 1;
    }
#if defined(NAM_BAKED) && NAM_HAS_LSTM
    if (!NAM_MODELS[m].a2) {
        RUN[NAM_MODELS[m].shape](s, &NAM_MODELS[m], buf, k);
        return 1;
    }
#endif
    return 0;
}

#ifndef OP1_HOST
volatile uint32_t nam_prof[3];          /* T1 monitor (screens/sysmon.c): cycles in the FX, blocks the model ran, blocks */
volatile uint32_t nam_synth_stamp;      /* op1_audio_blocks when "nam input" last replaced the synth bus (dispatcher.c) */
#endif
void nam_process(int pos, int16_t *buf, const int16_t *p, int input)
{
#ifndef OP1_HOST
    uint32_t t0, t1;
    asm volatile("%0 = CYCLES;" : "=d"(t0));
    if (input && pos == 0) nam_synth_stamp = op1_audio_blocks;
    int ran = nam_block(pos, buf, p, input);
    asm volatile("%0 = CYCLES;" : "=d"(t1));
    nam_prof[0] += t1 - t0;
    nam_prof[1] += (uint32_t)ran;
    nam_prof[2]++;
#else
    nam_block(pos, buf, p, input);
#endif
}

static void nam_render(int pos, int16_t *buf, const int16_t *p) { nam_process(pos, buf, p, 0); }
#ifdef NAM_BAKED
const op1_fx_t FX_nam = { nam_init, nam_render,
    { OP1_PRESS_NONE, OP1_PRESS_NONE, NAM_NUM_MODELS > 1 ? OP1_PRESS_PAGE : OP1_PRESS_NONE, OP1_PRESS_NONE } };
#else
const op1_fx_t FX_nam = { nam_init, nam_render,
    { OP1_PRESS_STEP, OP1_PRESS_NONE, OP1_PRESS_NONE, OP1_PRESS_NONE } };
#endif
