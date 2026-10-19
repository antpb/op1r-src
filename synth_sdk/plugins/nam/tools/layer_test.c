/* Blocks through each A2 layer of model 0, from pseudo-random rings and head sums; each layer's write position
 * advances as in run_a2, so the blocks cross every ring's wrap. Prints the entries each layer writes into the next
 * ring and every hs word, then whole run_a2 blocks (RUN_TEST). Built for the Blackfin simulator and natively by
 * tools/layer_test.sh, which diffs the two: locates a mismatch between the assembly kernel and the C reference. */
#include "op1_plugin.h"
void *host_fx_scratch[3];
static int32_t scratch[OP1_FX_SCRATCH_BYTES / 4];
#include "nam.c"
#ifdef BENCH_NATIVE
#include <stdio.h>
static void out(const char *s) { fputs(s, stdout); }
#else
static int sys(int nr, void *arg)
{
    register int r0 asm("R0") = (int)arg;
    register int p0 asm("P0") = nr;
    asm volatile("EXCPT 0;" : "+d"(r0) : "a"(p0) : "R1", "R2", "CC", "memory");
    return r0;
}
static int slen(const char *s) { int n = 0; while (s[n]) n++; return n; }
static void out(const char *s) { int blk[3] = { 1, (int)s, slen(s) }; sys(5, blk); }
#endif
static void outn(int32_t v)
{
    char b[14]; int i = 13, neg = v < 0; uint32_t u = neg ? -(uint32_t)v : (uint32_t)v; b[i] = 0;
    do { b[--i] = (char)('0' + u % 10); u /= 10; } while (u);
    if (neg) b[--i] = '-';
    out(b + i);
}
int main(void)
{
    for (int i = 0; i < 3; i++) host_fx_scratch[i] = scratch;
    static nam_state_t s;
    load_model(&s, 0, 0);
    const nam_a2_t *A = baked_a2[0];
    if (!A) { out("model 0 is not an A2 model\n"); return 1; }
    int16_t *ring = (int16_t *)scratch;
    int16_t *sink = (int16_t *)((uint8_t *)ring + NAM_A2_SINK_OFF);
    int32_t *hs = (int32_t *)((uint8_t *)ring + NAM_A2_HS_OFF);
    while (!fill_a2(&s, A, ring)) {}
    uint32_t seed = 1;
    for (int i = 0; i < A->ring * 4; i++) { seed = seed * 1664525u + 1013904223u; ring[i] = (int16_t)(seed >> 20) - 2048; }
    for (int i = 0; i < NAM_A2_BLOCK * 3; i++) { seed = seed * 1664525u + 1013904223u; hs[i] = (int32_t)(seed >> 8) - (1 << 23); }
    for (int blk = 0; blk < BLOCKS; blk++)
    for (int l = 0; l < LAYERS; l++) {
        const nam_a2_layer_t *L = &A->L[l];
        int16_t *nrg = l < 22 ? ring + L[1].off * 4 : sink;
        int nlen = l < 22 ? L[1].len : NAM_A2_BLOCK, nw = l < 22 ? s.w[l + 1] : 0;
        layer_a2(L, ring + L->off * 4, s.w[l], nrg, nlen, nw, hs);
        int w = s.w[l] + NAM_A2_BLOCK;
        s.w[l] = (uint16_t)(w >= L->len ? w - L->len : w);
        out("block "); outn(blk); out(" layer "); outn(l); out("\n");
        for (int f = 0; f < NAM_A2_BLOCK; f++) {
            int i = nw + f;
            const int16_t *o = nrg + (i >= nlen ? i - nlen : i) * 4;
            for (int c = 0; c < 4; c++) { outn(o[c]); out(" "); }
            for (int c = 0; c < 3; c++) { outn(hs[f * 3 + c]); out(" "); }
            out("\n");
        }
    }
#ifdef RUN_TEST
    static nam_state_t s2;
    s2.fade = FADE_LEN;
    while (!fill_a2(&s2, A, ring)) {}
    nam_knobs_t kk = { 4096, 32767, 16384 };
    int16_t buf[OP1_FX_BLOCK];
    for (int blk = 0; blk < 8; blk++) {
        for (int n = 0; n < OP1_FX_BLOCK; n++) { seed = seed * 1664525u + 1013904223u; buf[n] = (int16_t)(seed >> 18) - 8192; }
        run_a2(&s2, A, buf, kk, ring);
        out("run block "); outn(blk); out("\n");
        for (int n = 0; n < OP1_FX_BLOCK; n++) { outn(buf[n]); out(" "); }
        out("\n");
    }
#endif
    return 0;
}
