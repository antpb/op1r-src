/* "nam input" - the nam amp (nam.c) on the OP-1's live audio input. The amp runs once the mic/input
 * key is monitoring the line jack. The built-in mic is ignored. Profiles are 1.nam, 2.nam, ... in the nam
 * folder on the OP-1 disk; the blue encoder's press steps to the next one. The number on screen is that file.
 * Copyright (C) 2026 antpb. GPL-3.0-or-later with the OP-1 firmware permission (see LICENSE and NOTICE).
 */
#include "op1_plugin.h"
#include "seq/seq_text.h"
void nam_init(int pos);
void nam_process(int pos, int16_t *buf, const int16_t *p, int input);
int nam_file_num(int pos);
int nam_file_state(int pos);

static void naminput_render(int pos, int16_t *buf, const int16_t *p) { nam_process(pos, buf, p, 1); }
const op1_fx_t FX_naminput = { nam_init, naminput_render,
                               { OP1_PRESS_STEP, OP1_PRESS_NONE, OP1_PRESS_NONE, OP1_PRESS_NONE } };

#pragma GCC push_options
#pragma GCC optimize("Os")
static const uint8_t C_BLUE[4] = { 0x4D, 0x9E, 0xFF, 0xFF }, C_GREY[4] = { 0x6F, 0x71, 0x73, 0xFF },
                     C_WHITE[4] = { 0xFF, 0xFF, 0xFF, 0xFF }, C_RED[4] = { 0xFF, 0x3A, 0x5D, 0xFF };
static const uint8_t *const KC[4] = { C_BLUE, C_WHITE, C_WHITE, C_RED };
static const int32_t KX[4] = { 24, 100, 176, 252 };

static int32_t text_w(const char *t, int32_t s)
{
    int32_t w = 0;
    for (; *t; t++) w += (*t == '1' || *t == '.' ? 4 : 6) * s;
    return w - 2 * s;
}

static void naminput_screen_init(void *scr) { (void)scr; }

static void naminput_screen_update(void *scr, int dt)
{
    int pos = op1_fx_screen_pos() % 3;
    int num = nam_file_num(pos), st = nam_file_state(pos);
    op1_screen_draw(scr, dt);
    void *r = op1_raster();
    const uint8_t *nc = st == 3 ? C_RED : (st == 2 ? C_BLUE : C_GREY);
    char t[4];
    if (num > 0) sq_num(t, (uint32_t)num, 1);
    else { t[0] = '-'; t[1] = 0; }
    int32_t s = OP1_Q10(8);
    sq_text(r, OP1_Q10(160) - text_w(t, s) / 2, OP1_Q10(28), s, t, 0x800, nc);
    sq_text(r, OP1_Q10(16), OP1_Q10(12), OP1_Q10(2), "NAM", 0x600, C_GREY);
    int16_t k[4];
    sq_fx_knobs_live(k);
    static const char *const lab[4] = { "GAIN", "TONE", "GATE", "LEVL" };
    for (int i = 0; i < 4; i++) {
        int32_t x = OP1_Q10(KX[i]);
        sq_text(r, x, OP1_Q10(108), OP1_Q10(1), lab[i], 0x500, i == 0 ? C_BLUE : C_GREY);
        int32_t len = OP1_Q10(64) * (k[i] < 0 ? 0 : k[i]) >> 15;
        if (len > 0x200) sq_line(r, x, OP1_Q10(128), x + len, OP1_Q10(128), 0x500, KC[i]);
        sq_line(r, x + len, OP1_Q10(123), x + len, OP1_Q10(133), 0x500, KC[i]);
    }
}
#pragma GCC pop_options
const op1_screen_t SCREEN_naminput = { naminput_screen_init, naminput_screen_update };
