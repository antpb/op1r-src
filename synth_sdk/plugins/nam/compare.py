#!/usr/bin/env python3
"""Compare nam.c (fixed point, as on the OP-1) with NAM core (float) on the same audio, for every baked model.

    python3 plugins/nam/compare.py --ref <namraw> [--di guitar_di.wav]
    python3 plugins/nam/compare.py                                       # A2 bit-exactness only

<namraw>: NAM core reference renderer, built from tools/namraw.cpp (see README). Both sides get identical samples
(no resampling); the reference output gets the plugin's normalisation gain, DC blocker and clip in float, so the
difference is the fixed-point error alone. A2 models: NAM core runs the 3-channel submodel extracted from the file,
and nam.c's output must also be bit-exact with a2model.run_int (the integer reference the formats were designed
on). Signals: OP-1-like saw chords and plucks, a sine sweep, and a guitar DI (optional), each at 0 / +6 / +12 dB
drive. Metric: SNR of plugin vs reference (first 1024 samples skipped: fade-in).
Pass: SNR >= 40 dB (LSTM) / 28 dB (A2: int16 activations through 23 layers, see README), A2 bit-exact.
"""
import argparse, json, os, subprocess, sys, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SDK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import gen_model  # noqa: E402
import a2model  # noqa: E402

DRIVER = r'''
#include <stdio.h>
#include <stdlib.h>
#include "op1_plugin.h"
void *host_fx_scratch[3];
extern const op1_fx_t FX_nam;
int main(int argc, char **argv) {
    static int32_t scratch[OP1_FX_SCRATCH_BYTES / 4];
    for (int i = 0; i < 3; i++) host_fx_scratch[i] = scratch;
    int16_t k[8] = {0}, buf[OP1_FX_BLOCK];
    for (int i = 0; i < 4; i++) k[i] = (int16_t)atoi(argv[1 + i]);
    k[7] = (int16_t)atoi(argv[5]);                       /* white page 2: model */
    k[4] = OP1_ENGINE_MAGIC + 8;
    FX_nam.init(0);
    while (fread(buf, 2, OP1_FX_BLOCK, stdin) == OP1_FX_BLOCK) { FX_nam.render(0, buf, k); fwrite(buf, 2, OP1_FX_BLOCK, stdout); }
    return 0;
}
'''
SR = 44100
DRIVES = [(16384, 0), (21846, 1), (27307, 2)]      # knob words that land exactly on 0 / +6 / +12 dB (whole octaves)


def signals(di):
    t = np.arange(int(3 * SR)) / SR
    rng = np.random.default_rng(1)
    saw = lambda f: 2 * ((t * f) % 1) - 1
    env = np.exp(-((t * 2) % 1) * 4)                                           # plucks every 0.5 s
    chord = sum(saw(f) for f in (110, 138.6, 164.8, 220)) / 4 * env * 0.5      # -6 dBFS-ish OP-1 chord
    pluck = saw(82.4 * 2 ** (rng.integers(0, 24, len(t) // 22050 + 1)[(t * 2).astype(int)] / 12)) * env * 0.7
    f = 40 * (10000 / 40) ** (t / t[-1])
    sweep = 0.25 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    out = {'chord': chord, 'pluck': pluck, 'sweep': sweep}
    if di:
        w = wave.open(di)
        n, sw = w.getnframes(), w.getsampwidth()
        raw = np.frombuffer(w.readframes(min(n, 8 * w.getframerate())), np.uint8)
        if sw == 3:
            a = raw.reshape(-1, 3).astype(np.int32)
            x = ((a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)) << 8 >> 8) / 2 ** 23
        else:
            x = np.frombuffer(raw.tobytes(), np.int16) / 32768
        x = x.reshape(-1, w.getnchannels())[:, 0]
        out['guitar DI'] = x / np.abs(x).max() * 0.5
    return {k: np.round(np.clip(v, -1, 32767 / 32768) * 32768).astype(np.int16) for k, v in out.items()}


def post_int(y):
    """nam.c's post() at tone open, level 23170, fade-in from a fresh model load: in Python integers"""
    out, dc = np.empty(len(y), np.int64), 0
    for i, v in enumerate(int(v) for v in y):
        d = dc >> 10
        dc += v - d
        z = max(-32768, min(32767, v - d))
        o = (z * (23170 * 23170 >> 15)) >> 14
        if i < 256:
            o = (o * i) >> 8
        out[i] = max(-32768, min(32767, o))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref')
    ap.add_argument('--di')
    a = ap.parse_args()
    build = os.path.join(SDK, 'harness', 'build')
    os.makedirs(build, exist_ok=True)
    open(os.path.join(build, 'nam_driver.c'), 'w').write(DRIVER)
    exe = os.path.join(build, 'nam_compare')
    subprocess.run(['cc', '-O2', '-Wall', '-fsanitize=address,undefined', '-fno-sanitize-recover=undefined',
                    '-DOP1_HOST', '-DNAM_BAKED', '-I', SDK, '-I', HERE, os.path.join(build, 'nam_driver.c'),
                    os.path.join(HERE, 'nam.c'), '-o', exe], check=True)
    paths = sorted(os.path.join(HERE, 'models', f) for f in os.listdir(os.path.join(HERE, 'models')) if f.endswith('.nam'))
    sigs = signals(a.di)
    worst, exact = {'lstm': 99.0, 'a2': 99.0}, True
    for mi, path in enumerate(paths):
        m = gen_model.load(path)
        norm = gen_model.norm_gain(m)
        km = (mi * 32768 + 16384) // len(paths)
        ref_path, Q, skip = path, None, 0
        if m['kind'] == 'a2':
            ref_path = os.path.join(build, f'nam_ref_{mi}.nam')
            json.dump(m['m'], open(ref_path, 'w'))
            Q = a2model.convert(m['m'], norm)
            ring = sum(a2model.ring_size(K, D) for K, D in zip(a2model.KS, a2model.DS))
            skip = (-(-ring // 2048) - 1) * 64                  # nam.c fills its rings first (FILL_CHUNK per block)
            print(f'{os.path.basename(path)} ({m["sr"]} Hz model, A2 nano WaveNet):')
        else:
            print(f'{os.path.basename(path)} ({m["sr"]} Hz model, LSTM {len(m["layers"])}x{m["H"]}):')
        for name, x in sigs.items():
            row = []
            for kg, octv in DRIVES:
                raw = np.frombuffer(subprocess.run([exe, str(kg), '32767', '0', '23170', str(km)], input=x.tobytes(),
                                                   capture_output=True, check=True).stdout, np.int16)
                got = raw / 32768
                if Q is not None:                                   # bit-exact with the integer reference
                    xq = np.clip((x.astype(np.int64) * (4096 << octv)) >> 15, -32768, 32767)
                    want = np.concatenate([np.zeros(skip, np.int64), post_int(a2model.run_int(Q, xq[skip:]))])
                    want = want[:len(raw)]
                    if not np.array_equal(want, raw.astype(np.int64)):
                        exact = False
                        print(f'  NOT BIT-EXACT: {name} drive {octv}: first diff at {np.argmax(want != raw)}')
                if not a.ref:
                    if Q is not None:
                        row.append(f'{["0", "+6", "+12"][octv]:>3} dB: {"bit-exact" if np.array_equal(want, raw) else "DIFF"}')
                    continue
                xin = (x.astype(np.float32) / 32768 * 2 ** octv).astype(np.float32)
                xin[:skip] = 0                                      # the plugin ignores input while filling
                y = np.frombuffer(
                    subprocess.run([a.ref, ref_path], input=xin.tobytes(), capture_output=True, check=True).stdout,
                    np.float32).astype(np.float64)[:len(got)] * norm
                dc, ref = 0.0, np.empty_like(y)
                for i, v in enumerate(y):                           # plugin's DC blocker, in float
                    d = dc / 1024; dc += v - d; ref[i] = v - d
                ref = np.clip(ref, -1, 1) * (23170 * 23170 >> 15) / 16384
                e, r = got[skip + 1024:] - ref[skip + 1024:], ref[skip + 1024:]
                snr = 10 * np.log10(np.sum(r * r) / max(np.sum(e * e), 1e-20))
                worst[m['kind']] = min(worst[m['kind']], snr)
                row.append(f'{["0", "+6", "+12"][octv]:>3} dB: SNR {snr:5.1f} dB (peak {20 * np.log10(np.abs(r).max() + 1e-9):5.1f} dBFS)')
            print(f'  {name:10s} ' + '   '.join(row))
    ok = (not a.ref or (worst['lstm'] >= 40 and worst['a2'] >= 28)) and exact
    fmt = lambda k: 'n/a' if worst[k] == 99.0 else f'{worst[k]:.1f} dB'
    print(f'worst SNR: LSTM {fmt("lstm")}, A2 {fmt("a2")}; A2 bit-exact: {exact}: ' +
          ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
