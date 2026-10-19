/* Resource monitor page (docs/18): clock screen (Shift+help) + T1. Shows CPU (total), AUDIO (audio callback),
 * RAM (both heaps), DISK (yaffs) and MIDI ring fill. Geometry/colours: sysmon_geom.h (display/gen_monitor_svg.py).
 *
 * Data sources (filled by hooks in dispatcher.c / hook.S):
 *   sm_audio_cb      wraps the audio DMA callback 0xFFA04804 (ISR): busy cycles per block
 *   sm_idle_run      wraps kTE_Idle::run: remembers the idle thread object -> VDK run-cycle stats
 * Everything else is read in the GUI thread, a few times per second (frame counter, no clock needed: all loads
 * are ratios of CYCLES).
 */
#include "../op1_synth.h"
#include "sysmon_geom.h"

#ifdef OP1_HOST
extern uint32_t sm_host_cycles, sm_host_idle_lo;            /* harness: fake counters / sources */
extern int sm_host_heap(int i, uint32_t *size, uint32_t *used);
extern int sm_host_disk(uint32_t *total, uint32_t *free_);
extern uint32_t sm_host_midi_fill[2];
#define READ_CYCLES()            (sm_host_cycles)
#define IDLE_TOTAL_LO(stats)     (sm_host_idle_lo)
#define AUDIO_CB(a, b, c)        ((void)0)
#else
static inline uint32_t READ_CYCLES(void) { uint32_t c; __asm__ volatile ("%0 = CYCLES;" : "=d"(c)); return c; }
#define IDLE_TOTAL_LO(stats)     (((volatile uint32_t *)(stats))[0x20 / 4])   /* VDK stats: total run cycles (lo) */
#define AUDIO_CB(a, b, c)        (((void (*)(void *, void *, void *))0xFFA04804)((a), (b), (c)))
#define HEAP_LOOKUP(i)           (((uint32_t *(*)(int))0x01A9F2E2)(i))
#define LOCK_SCHED()             (((void (*)(void))0x01A9F76C)())
#define UNLOCK_SCHED()           (((void (*)(void))0x01A9F78E)())
#define YAFFS_FREE(p)            (((int32_t (*)(const char *))0x01A9B1C0)(p))
#define YAFFS_TOTAL(p)           (((int32_t (*)(const char *))0x01A9B284)(p))
#define MIDI_RX_WR               (*(volatile uint32_t *)0x01AB1510)
#define MIDI_RX_RD               (*(volatile uint32_t *)0x01AB1514)
#define MIDI_TX_WR               (*(volatile uint32_t *)0x01AB2B44)
#define MIDI_TX_RD               (*(volatile uint32_t *)0x01AB1B3C)
#endif

/* ---------------- sources ---------------- */
static volatile uint32_t aud_busy, aud_max, aud_last_t0, aud_max_period;   /* ISR side, reset by the GUI */
/* Live input for plugins (op1_plugin.h op1_audio_input): the callback gets (out block, in block, ?), 64 frames of
 * (L, R) 32-bit words each (the firmware's audio test mode 0x0191A7A8 copies in -> out; its tone mode 0x0191A880
 * writes out = R0). The in block is copied here, before the firmware's render can touch it, when a plugin asks. */
volatile uint8_t op1_audio_in_on;
const void *volatile op1_audio_in_src;
int32_t op1_audio_in[128];
volatile uint32_t op1_audio_blocks;
void sm_audio_cb(void *a, void *b, void *c)                  /* replaces the callback pointer (audio::init) */
{
    uint32_t t0 = READ_CYCLES();
    op1_audio_blocks++;
    if (op1_audio_in_on && b) {
        const int32_t *in = (const int32_t *)b;
        for (int i = 0; i < 128; i++) op1_audio_in[i] = in[i];
        op1_audio_in_src = b;
    }
    AUDIO_CB(a, b, c);
    uint32_t d = READ_CYCLES() - t0, p = t0 - aud_last_t0;
    aud_last_t0 = t0;
    aud_busy += d;
    if (d > aud_max) { aud_max = d; aud_max_period = p; }
}

static void *volatile idle_obj;                              /* kTE_Idle thread object (VDK thread) */
#ifndef OP1_HOST
void sm_idle_run(void *self)                                 /* kTE_Idle vtable run slot; never returns */
{
    idle_obj = self;
    ((void (*)(void *))0x01A5E240)(self);
}
#endif
static void *idle_stats(void)                                /* VDK stats block (heap pointer at thread+0x34) */
{
#ifdef OP1_HOST
    return (void *)1;
#else
    uint32_t s = idle_obj ? ((volatile uint32_t *)idle_obj)[0x34 / 4] : 0;
    return (s >= 0x100 && s < 0x04000000 && !(s & 3)) ? (void *)s : 0;   /* SDRAM heap pointer, else unknown */
#endif
}

/* optional profile line: paulstretch phase cycles (shelved/paulstretch/paulstretch_dsp.c), weak: absent = no line */
extern volatile uint32_t ps_prof[7] __attribute__((weak));
static uint32_t prof_last[7];
static int32_t prof_pct[7];
extern volatile uint32_t dp_prof[2] __attribute__((weak));   /* drum+ voice render cycles (drumplus/dp_fw.c) */
extern volatile uint32_t nam_diag[4] __attribute__((weak));  /* nam input source (plugins/nam): addr, word, low bits, resyncs */
extern volatile uint32_t nam_prof[3] __attribute__((weak));  /* nam FX cycles, blocks the model ran, blocks (plugins/nam) */
static uint32_t nd[4], np_last[3];
static int32_t np_pct, np_pk, np_run;
static uint32_t dp_last[2];
static int32_t dp_pct[2], dp_pk[2];

/* ---------------- state ---------------- */
#define PEAK_DECAY 2                                        /* peak falls by 2 % per sample (~4 samples/s) */
static int32_t v_pct[SM_ROWS], pk_pct[SM_ROWS], midi_pct[2], midi_pk[2];
static uint32_t ram_used, ram_size, disk_used, disk_size;
static uint32_t last_wall, last_idle, frame;
static int have_cpu;

static int32_t pct(uint32_t a, uint32_t b)                   /* 100 * a / b without overflow / division by 0 */
{
    if (!b) return 0;
    while (a > 0x7FFFFFFFu / 100) { a >>= 1; b >>= 1; }
    if (!b) return 100;
    int32_t r = (int32_t)(a * 100 / b);
    return r > 100 ? 100 : r;
}
static void peak(int32_t *pk, int32_t v) { *pk = v > *pk - PEAK_DECAY ? v : *pk - PEAK_DECAY; }

static void sample_fast(void)                                /* CPU, AUDIO, MIDI: every few frames */
{
    uint32_t now = READ_CYCLES(), wall = now - last_wall;
    void *st = idle_stats();
    if (st) {
        uint32_t idle = IDLE_TOTAL_LO(st), di = idle - last_idle;
        if (have_cpu) { v_pct[SM_CPU] = 100 - pct(di, wall); peak(&pk_pct[SM_CPU], v_pct[SM_CPU]); }
        last_idle = idle; have_cpu = 1;
    }
    if (ps_prof)                                             /* gather fft mags synth bitrev ifft out */
        for (int i = 0; i < 7; i++) { uint32_t c = ps_prof[i]; prof_pct[i] = pct(c - prof_last[i], wall); prof_last[i] = c; }
    if (dp_prof)
        for (int i = 0; i < 2; i++) { uint32_t c = dp_prof[i]; dp_pct[i] = pct(c - dp_last[i], wall); dp_last[i] = c; peak(&dp_pk[i], dp_pct[i]); }
    if (nam_diag) {                                          /* hold the largest word / bits since the last sample */
        nd[0] = nam_diag[0]; nd[1] = nam_diag[1]; nd[2] = nam_diag[2]; nd[3] = nam_diag[3];
        nam_diag[1] = 0; nam_diag[2] = 0; nam_diag[3] = 0;
    }
    if (nam_prof) {
        uint32_t c = nam_prof[0], ran = nam_prof[1], blk = nam_prof[2];
        np_pct = pct(c - np_last[0], wall);
        peak(&np_pk, np_pct);
        if (blk != np_last[2]) np_run = pct(ran - np_last[1], blk - np_last[2]);
        np_last[0] = c; np_last[1] = ran; np_last[2] = blk;
    }
    uint32_t busy = aud_busy, mx = aud_max, mp = aud_max_period;
    aud_busy = 0; aud_max = 0;
    v_pct[SM_DSP] = pct(busy, wall);
    peak(&pk_pct[SM_DSP], mp ? pct(mx, mp) : v_pct[SM_DSP]);
    last_wall = now;
#ifdef OP1_HOST
    uint32_t fill[2] = { sm_host_midi_fill[0], sm_host_midi_fill[1] };
#else
    uint32_t fill[2] = { (MIDI_RX_WR - MIDI_RX_RD) & 0x3FF, (MIDI_TX_WR - MIDI_TX_RD) & 0x3FF };   /* 1024 entries */
#endif
    for (int i = 0; i < 2; i++) { midi_pct[i] = pct(fill[i], 1024); peak(&midi_pk[i], midi_pct[i]); }
}

static void sample_ram(void)
{
    ram_used = ram_size = 0;
    for (int i = 0; i < 2; i++) {
#ifdef OP1_HOST
        uint32_t sz, used;
        if (!sm_host_heap(i, &sz, &used)) continue;
#else
        LOCK_SCHED();                                        /* the lock malloc/free use */
        uint32_t *h = HEAP_LOOKUP(i);
        if (!h) { UNLOCK_SCHED(); continue; }
        uint32_t sz = h[1], used = h[3] - h[0], fr = 0;      /* desc: base, size, ?, bump; free list at +0x18 */
        const uint32_t *b = (const uint32_t *)h[0x18 / 4];
        for (int n = 0; b && n < 8192; n++) { fr += b[0]; b = (const uint32_t *)b[2]; }   /* {size, heap, next} */
        UNLOCK_SCHED();
        used = used > fr ? used - fr : 0;
#endif
        ram_size += sz; ram_used += used;
    }
    v_pct[SM_RAM] = pct(ram_used, ram_size);
}

static void sample_disk(void)
{
#ifdef OP1_HOST
    uint32_t total, fr;
    if (!sm_host_disk(&total, &fr)) return;
#else
    /* shortcut: no check for USB disk mode (flag not found yet); upgrade if the page ever runs during disk mode */
    int32_t t = YAFFS_TOTAL("/yaffs2"), f = YAFFS_FREE("/yaffs2");
    if (t <= 0 || f < 0) return;
    uint32_t total = (uint32_t)t, fr = (uint32_t)f;
#endif
    disk_size = total; disk_used = total > fr ? total - fr : 0;
    v_pct[SM_DISK] = pct(disk_used, disk_size);
}

/* ---------------- text ---------------- */
static char *put_u(char *p, uint32_t v)                     /* decimal, no libc */
{
    char t[10]; int n = 0;
    do { t[n++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (n) *p++ = t[--n];
    return p;
}
static char *put_s(char *p, const char *s) { while (*s) *p++ = *s++; return p; }
static char *put_mb(char *p, uint32_t bytes, int dec)       /* firmware font: no uppercase glyphs (they render as symbols); "13.1" (dec) or "171" MB, 1 MB = 2^20 */
{
    uint32_t t = (bytes >> 10) * 10 / 1024;                  /* tenths of MB */
    p = put_u(p, t / 10);
    if (dec) { *p++ = '.'; *p++ = (char)('0' + t % 10); }
    return p;
}
static void value_text(int row, char *o)
{
    char *p = o;
    switch (row) {
    case SM_CPU:  if (have_cpu) { p = put_u(p, (uint32_t)v_pct[row]); *p++ = '%'; } else p = put_s(p, "--"); break;
    case SM_DSP:  p = put_u(p, (uint32_t)v_pct[row]); *p++ = '%'; break;
    case SM_RAM:  p = put_mb(p, ram_used, 1); *p++ = '/'; p = put_mb(p, ram_size, 0); *p++ = 'm'; *p++ = 'b'; break;
    case SM_DISK: p = put_mb(p, disk_used, 0); *p++ = '/'; p = put_mb(p, disk_size, 0); *p++ = 'm'; *p++ = 'b'; break;
    case SM_MIDI: p = put_u(p, (uint32_t)midi_pk[0]); *p++ = '/'; p = put_u(p, (uint32_t)midi_pk[1]); *p++ = '%'; break;
    }
    *p = 0;
}

/* ---------------- screen ---------------- */
#define LABEL_X   8
#define VALUE_X   (SM_BX1 + 6)
#define TEXT_SIZE 12
static uint32_t txt_mem[2 * SM_ROWS + 3][OP1_TEXT_SIZE / 4];    /* label + value per row, "in" / "out", profile */
static int txt_ok;
static const int32_t ident[9] = { 1024, 0, 0, 0, 1024, 0, 0, 0, 1024 };
static const uint8_t c_label[4] = { 0xAE, 0xB1, 0xDC, 0xFF }, c_value[4] = { 0xFF, 0xFF, 0xFF, 0xFF },
                     c_dim[4] = { 0x6F, 0x71, 0x73, 0xFF }, c_red[4] = { 0xFF, 0x3A, 0x5D, 0xFF },
                     c_white[4] = { 0xFF, 0xFF, 0xFF, 0xFF };

static void sm_init(void *scr)
{
    (void)scr;
    for (int r = 0; r < SM_ROWS; r++) {
        op1_text_ctor(txt_mem[2 * r], OP1_Q10(LABEL_X), sm_row_y[r], TEXT_SIZE, c_label);
        op1_text_set(txt_mem[2 * r], sm_row_label[r]);
        op1_text_ctor(txt_mem[2 * r + 1], OP1_Q10(VALUE_X), sm_row_y[r], TEXT_SIZE, c_value);
    }
    op1_text_ctor(txt_mem[2 * SM_ROWS], OP1_Q10(SM_BX0 + 2), sm_row_y[SM_MIDI] + 14, TEXT_SIZE, c_dim);
    op1_text_set(txt_mem[2 * SM_ROWS], "in");
    op1_text_ctor(txt_mem[2 * SM_ROWS + 1], OP1_Q10(SM_MID + 6), sm_row_y[SM_MIDI] + 14, TEXT_SIZE, c_dim);
    op1_text_set(txt_mem[2 * SM_ROWS + 1], "out");
    op1_text_ctor(txt_mem[2 * SM_ROWS + 2], OP1_Q10(LABEL_X), 136, TEXT_SIZE, c_dim);
    txt_ok = 1;
    sample_ram(); sample_disk();
}

static void bar(void *r, int x0, int x1, int y, int32_t v, int32_t pk, const uint8_t *rgb, int show_pk)
{
    uint8_t c[4] = { rgb[0], rgb[1], rgb[2], 0xFF };
    int32_t w = (x1 - x0 - 2) * v / 100;                    /* one thick line per bar */
    if (w >= 1) {
        op1_pt a = { OP1_Q10(x0 + 1), OP1_Q10(y) + OP1_Q10(SM_BH) / 2 }, b = { OP1_Q10(x0 + 1 + w), a.y };
        op1_raster_line(r, &a, &b, OP1_Q10(SM_BH - 2), c);
    }
    if (show_pk) {                                           /* peak marker: white, red at the top */
        int32_t px = OP1_Q10(x0 + 1) + OP1_Q10(x1 - x0 - 2) * pk / 100;
        op1_pt a = { px, OP1_Q10(y - 2) }, b = { px, OP1_Q10(y + SM_BH + 2) };
        op1_raster_line(r, &a, &b, OP1_Q10(1.5), pk >= 98 ? c_red : c_white);
    }
}

static void sm_update(void *scr, int dt)
{
    if ((frame & 7) == 0) sample_fast();
    if ((frame & 31) == 16) sample_ram();
    if ((frame & 255) == 128) sample_disk();
    frame++;
    op1_screen_draw(scr, dt);
    void *r = op1_raster();
    for (int row = 0; row < SM_ROWS; row++) {
        int y = sm_row_y[row];
        if (row == SM_MIDI) {
            bar(r, SM_BX0, SM_MID - 4, y, midi_pct[0], midi_pk[0], sm_row_rgb[row], 1);
            bar(r, SM_MID + 4, SM_BX1, y, midi_pct[1], midi_pk[1], sm_row_rgb[row], 1);
        } else
            bar(r, SM_BX0, SM_BX1, y, v_pct[row], pk_pct[row], sm_row_rgb[row], row <= SM_DSP);
    }
    if (!txt_ok) return;
    static char last[SM_ROWS][24];
    for (int row = 0; row < SM_ROWS; row++) {
        char s[24]; int k = 0;
        value_text(row, s);
        while (s[k] && s[k] == last[row][k]) k++;
        if (s[k] != last[row][k]) {                          /* text_set rebuilds the glyph lists: only on change */
            op1_text_set(txt_mem[2 * row + 1], s);
            for (k = 0; (last[row][k] = s[k]) != 0; k++) ;
        }
        op1_text_draw(txt_mem[2 * row], r, ident);
        op1_text_draw(txt_mem[2 * row + 1], r, ident);
    }
    op1_text_draw(txt_mem[2 * SM_ROWS], r, ident);
    op1_text_draw(txt_mem[2 * SM_ROWS + 1], r, ident);
    extern int32_t op1_db_last_err __attribute__((weak));
    int dp_line = !ps_prof && dp_prof && (dp_pk[0] || dp_pk[1]);
    if (dp_line) {                                           /* "drum+ 9 stock 3" (% of the audio time, peaks) */
        char t[32], *q = put_s(t, "drum+ "); q = put_u(q, (uint32_t)dp_pk[0]);
        q = put_s(q, " stock "); q = put_u(q, (uint32_t)dp_pk[1]); *q = 0;
        static char lastd[32];
        int k = 0;
        while (t[k] && t[k] == lastd[k]) k++;
        if (t[k] != lastd[k]) { op1_text_set(txt_mem[2 * SM_ROWS + 2], t); for (k = 0; (lastd[k] = t[k]) != 0; k++) ; }
        op1_text_draw(txt_mem[2 * SM_ROWS + 2], r, ident);
    }
    if (!dp_line && !ps_prof && &op1_db_last_err && op1_db_last_err) {   /* a DB write failed (op1_db_write): "db err 7" */
        static int32_t shown = 0;
        if (shown != op1_db_last_err) {
            char t[16], *q = put_s(t, "db err "); q = put_u(q, (uint32_t)op1_db_last_err); *q = 0;
            op1_text_set(txt_mem[2 * SM_ROWS + 2], t);
            shown = op1_db_last_err;
        }
        op1_text_draw(txt_mem[2 * SM_ROWS + 2], r, ident);
    }
    /* "nam 21 pk30 run100 a370 s34": % of the audio time (now, peak), % of blocks the model ran (the rest slept),
     * then the input block address (hex) and resyncs when nam input is live */
    if (!dp_line && !ps_prof && ((nam_diag && nd[0]) || (nam_prof && np_pk))) {
        static const char hx[] = "0123456789abcdef";
        char t[48], *q = put_s(t, "nam ");
        q = put_u(q, (uint32_t)np_pct);
        q = put_s(q, " pk"); q = put_u(q, (uint32_t)np_pk);
        q = put_s(q, " run"); q = put_u(q, (uint32_t)np_run);
        if (nam_diag && nd[0]) {
            q = put_s(q, " a");
            for (int i = 8; i >= 0; i -= 4) *q++ = hx[(nd[0] >> i) & 15];
            q = put_s(q, " s"); q = put_u(q, nd[3]);
        }
        *q = 0;
        static char lastn[48];
        int k = 0;
        while (t[k] && t[k] == lastn[k]) k++;
        if (t[k] != lastn[k]) { op1_text_set(txt_mem[2 * SM_ROWS + 2], t); for (k = 0; (lastn[k] = t[k]) != 0; k++) ; }
        op1_text_draw(txt_mem[2 * SM_ROWS + 2], r, ident);
    }
    if (ps_prof) {                                           /* "paul 41 f12 m5 s9 b1 i12 o2" (% of the audio time) */
        static const char tag[7] = { 'g', 'f', 'm', 's', 'b', 'i', 'o' };
        char t[48], *q = put_s(t, "paul ");
        int32_t sum = 0;
        for (int i = 0; i < 7; i++) sum += prof_pct[i];
        q = put_u(q, (uint32_t)sum);
        for (int i = 0; i < 7; i++) { *q++ = ' '; *q++ = tag[i]; q = put_u(q, (uint32_t)prof_pct[i]); }
        *q = 0;
        static char lastp[48];
        int k = 0;
        while (t[k] && t[k] == lastp[k]) k++;
        if (t[k] != lastp[k]) { op1_text_set(txt_mem[2 * SM_ROWS + 2], t); for (k = 0; (lastp[k] = t[k]) != 0; k++) ; }
        op1_text_draw(txt_mem[2 * SM_ROWS + 2], r, ident);
    }
}

const op1_screen_t SCREEN_sysmon = { sm_init, sm_update };
