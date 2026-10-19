/* Cycle count of nam.c render() on the gdb Blackfin simulator (bfin-elf-run), per model in nam_models.h.
 * The simulator counts instructions with Blackfin's issue rules but has no memory wait states, so real OP-1 numbers
 * are a little higher (tables in SDRAM through the cache). Build + run: tools/bench.sh.
 * Each line ends with a hash of the output; bench.sh also builds this file natively (BENCH_NATIVE, plain C dot
 * product) and the hashes must match: the Blackfin assembly path is bit-exact. After the timed tone comes a gap of
 * silence longer than the receptive field and more tone: the "rest" column is the cost of a block while the model
 * sleeps, and bench.sh checks the hash against a native build that never sleeps (NAM_NO_REST). */
#include "op1_plugin.h"
extern const op1_fx_t FX_nam;
void *host_fx_scratch[3];
static int32_t scratch[OP1_FX_SCRATCH_BYTES / 4];
#define NAM_BENCH_NUM_MODELS NAM_NUM_MODELS_FROM_HEADER

#ifdef BENCH_NATIVE
#include <stdio.h>
static void out(const char *s) { fputs(s, stdout); }
static inline uint32_t cycles(void) { return 0; }
#else
static int sys(int nr, void *arg)
{
    register int r0 asm("R0") = (int)arg;
    register int p0 asm("P0") = nr;
    asm volatile("EXCPT 0;" : "+d"(r0) : "a"(p0) : "R1", "R2", "CC", "memory");   /* R1, R2: 2nd result, errno */
    return r0;
}
static int slen(const char *s) { int n = 0; while (s[n]) n++; return n; }
static void out(const char *s) { int blk[3] = { 1, (int)s, slen(s) }; sys(5, blk); }
static inline uint32_t cycles(void) { uint32_t c; asm volatile("%0 = CYCLES;" : "=d"(c)); return c; }
#endif
static void outn(uint32_t v)
{
    char b[12]; int i = 11; b[i] = 0;
    do { b[--i] = (char)('0' + v % 10); v /= 10; } while (v);
    out(b + i);
}

int main(void)
{
    int16_t k[8] = { 16384, 20000, 0, 23170, OP1_ENGINE_MAGIC + 8, 0, 0, 0 }, buf[OP1_FX_BLOCK];
    uint32_t ph = 0;
    for (int i = 0; i < 3; i++) host_fx_scratch[i] = scratch;
    for (int m = 0; m < NAM_BENCH_NUM_MODELS; m++) {
        k[7] = (int16_t)((m * 32768 + 16384) / NAM_BENCH_NUM_MODELS);   /* white page 2: model (gate off) */
        FX_nam.init(0);
        uint32_t total = 0, rest = 0, hash = 2166136261u;
        for (int blk = 0; blk < 42 + 120 + 20; blk++) {
            int quiet = blk >= 42 && blk < 42 + 120;
            for (int n = 0; n < OP1_FX_BLOCK; n++) { ph += 11000000u; buf[n] = quiet ? 0 : (int16_t)((int32_t)ph >> 17); }
            uint32_t t0 = cycles();
            FX_nam.render(0, buf, k);
            uint32_t t1 = cycles();
            if (blk >= 10 && blk < 42) total += t1 - t0;     /* skip model load + A2 ring fill (4 blocks) */
            if (blk == 42 + 119) rest = t1 - t0;
            for (int n = 0; n < OP1_FX_BLOCK; n++) hash = (hash ^ (uint16_t)buf[n]) * 16777619u;
#ifdef BENCH_TRACE
            out("blk "); outn((uint32_t)blk); out(" "); outn(hash); out(" "); outn((uint16_t)buf[0]); out("\n");
#endif
        }
        uint32_t per_sample = total / (32 * OP1_FX_BLOCK);
        out("model "); outn((uint32_t)m); out(": "); outn(per_sample); out(" cycles/sample, ");
        outn(per_sample * 44100u / 400000u); out(" per mille of 400 MHz, rest "); outn(rest / OP1_FX_BLOCK);
        out(" cycles/sample, hash "); outn(hash); out("\n");
    }
    return 0;
}
