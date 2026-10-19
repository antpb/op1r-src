#!/usr/bin/env python3
"""Generate nam_models.h and the device packs plugins/nam/packs/1.nam, 2.nam, ...

    python3 plugins/nam/gen_model.py                    # models/*.nam, sorted by file name
    python3 plugins/nam/gen_model.py a.nam b.nam        # these, in this order -> packs/1.nam, packs/2.nam
    python3 plugins/nam/gen_model.py --packs DIR a.nam  # write the packs into DIR (the OP-1's nam folder)
    python3 plugins/nam/gen_model.py -o x.h a.nam       # header elsewhere (benchmarks)

The packs are the fixed-point files the OP-1 loads. A JSON .nam from Neural Amp Modeler is not one of them.

Two architectures fit the OP-1: A2 nano WaveNets (3 channels; a2model.py has their fixed-point conversion) and LSTMs
(1..NAM_MAX_LAYERS layers, hidden size 1..NAM_MAX_HIDDEN, one input, one output). Standard WaveNet captures are far
too heavy (13k+ weights, ~15k multiplies per sample). The rest of this text is about LSTMs.

Math (same as NAM core lstm.cpp): per layer, xh = [x, h]; ifgo = W xh + b;
  c = sigmoid(f) c + sigmoid(i) tanh(g);  h = sigmoid(o) tanh(c);  y = head . h_last + head_b
Fixed point:
  - x, h: int16 Q15. The model input is scaled by 1/XS (x_q15 = x / XS) so the drive knob can push XS (+18 dB) into
    the model without clipping; the factor XS is folded into the input column of layer 0's W.
  - W: int16, rows interleaved by pairs (two rows per dual-MAC instruction in nam.c), QW fractional bits per layer; b: int32 at Q(QW + 15) (the accumulator's format). QW is the largest
    value with max|w| 2^QW <= 32767 and, per row, (sum|w| + |b|) 2^QW < 2^16, so W xh + b never overflows int32
    (|x|, |h| <= 1). Accumulator -> Q13 for the activation tables: >> (QW + 2).
  - c: int32 Q15, clamped to +-CMAX (tanh is flat there).
  - head: int16 (zero-padded to an even length >= 4), QH bits, same rule; the loudness normalisation gain (TARGET_DB - metadata loudness) is folded in.
  - Initial h, c: NAM resets to the state stored in the weights and then prewarms 0.5 s of silence. That steady
    state is computed here (float) and stored, so the plugin starts settled without running the prewarm.
Activation tables: 512 segments over [-8, 8] (step 1/32), Q15 value and delta packed in one word, linear
interpolation: max error ~1e-4. All arrays are 4-byte aligned (32-bit loads; a misaligned one faults the OP-1).
"""
import json, math, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a2model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
XS = 8.0                    # input headroom: model input = x_q15 / 32768 * XS
TARGET_DB = -18.0           # output loudness after normalisation (NAM plugin convention)
CMAX = 8.0                  # cell state clamp
MAX_LAYERS, MAX_HIDDEN = 2, 8
TAB_N = 512                 # segments over [-8, 8]


def sigmoid(v): return 1.0 / (1.0 + np.exp(-v))


def load(path):
    d = json.load(open(path))
    name = os.path.splitext(os.path.basename(path))[0]
    if d.get('architecture') != 'LSTM':
        m = a2model.pick(d)
        if m is None:
            sys.exit(f'{path}: architecture {d.get("architecture")!r}: the OP-1 runs LSTMs and A2 nano WaveNets '
                     f'(3 channels, or a SlimmableContainer with one)')
        meta = m.get('metadata') or d.get('metadata') or {}
        return dict(kind='a2', name=name, m=m, sr=m.get('sample_rate'), loudness=meta.get('loudness'))
    c = d['config']
    L, H, I = c['num_layers'], c['hidden_size'], c['input_size']
    if I != 1 or c.get('in_channels', 1) != 1 or c.get('out_channels', 1) != 1:
        sys.exit(f'{path}: needs 1 input / 1 output')
    if not (1 <= L <= MAX_LAYERS and 1 <= H <= MAX_HIDDEN):
        sys.exit(f'{path}: {L} layers x {H} hidden, max {MAX_LAYERS} x {MAX_HIDDEN} on the OP-1')
    w = iter(np.asarray(d['weights'], dtype=np.float64))
    take = lambda n: np.array([next(w) for _ in range(n)])
    layers = []
    for l in range(L):
        n_in = I if l == 0 else H
        W = take(4 * H * (n_in + H)).reshape(4 * H, n_in + H)
        layers.append(dict(W=W, b=take(4 * H), h=take(H), c=take(H), n_in=n_in))
    head, head_b = take(H), take(1)[0]
    assert next(w, None) is None, f'{path}: weight count mismatch'
    meta = d.get('metadata') or {}
    return dict(kind='lstm', name=name, layers=layers, head=head, head_b=head_b, H=H,
                sr=d.get('sample_rate'), loudness=meta.get('loudness'), meta=meta)


def norm_gain(m):
    """loudness normalisation to TARGET_DB (NAM plugin convention); 1 without metadata"""
    return 10 ** ((TARGET_DB - m['loudness']) / 20) if m['loudness'] is not None else 1.0


def a2_layer_weights(L):
    """the two weight streams layer_a2 reads (nam.c). W is [tap][in][out], l1 is [out][in]"""
    W, l1 = L['W'], L['l1']
    w01 = [int(v) for k_ in range(L['K']) for c in range(3) for v in (W[k_][c][0], W[k_][c][1])]
    w01 += [int(v) for v in L['mix'][:2]]
    w01 += [int(v) for j in range(3) for v in (l1[0][j], l1[1][j])]
    w2 = [int(v) for k_ in range(L['K']) for v in (W[k_][0][2], W[k_][1][2], W[k_][2][2], 0)]
    w2 += [int(L['mix'][2]), 0, int(l1[2][0]), int(l1[2][1]), int(l1[2][2]), 0]
    return w01, w2


def align4(b):
    return b + b'\x00' * ((4 - len(b)) % 4)


def legacy_ring(K, D):
    """the ring size written into packs: a power of 2, which loaders before exact rings require. nam.c ignores it."""
    return 1 << int(np.ceil(np.log2((K - 1) * D + a2model.BLOCK)))


def pack_bytes(m, path):
    """one A2 capture in the pack format nam.c parses (little endian, magic NA2): (bytes, nam.c ring entries)"""
    Q = a2model.convert(m['m'], norm_gain(m))
    layers = Q['layers']
    meta, ring, exact = [], 0, 0
    for L in layers:
        size = legacy_ring(int(L['K']), int(L['D']))
        w01, w2 = a2_layer_weights(L)
        meta.append((L, size, ring, w01, w2))
        ring += size
        exact += a2model.ring_size(int(L['K']), int(L['D']))
    b = bytearray(b'NA2\x01')
    b += struct.pack('<HH', ring, len(layers))
    b += struct.pack('<3h', *[int(v) for v in Q['rech']])
    b += struct.pack('<4B', int(Q['s0']), int(Q['eh'] - Q['ez_min']), int(Q['qh'] - Q['eh']), 0)
    b += struct.pack('<i', int(Q['hb']))
    b += struct.pack('<3h', *[int(v) for v in Q['steady']['h']])
    b += struct.pack('<H', 0)
    b = align4(b)
    head = [int(v) for v in np.ravel(Q['head'])]
    if len(head) != 48:
        sys.exit(f'{path}: head {len(head)}')
    b += struct.pack('<' + 'h' * 48, *head)
    for i, (L, size, off, w01, w2) in enumerate(meta):
        b = align4(b)
        b += struct.pack('<6B', int(L['K']), int(L['sz']), int(L['sr']), int(L['s_in']), int(L['sh']), 0)
        b += struct.pack('<3H', int(L['D']), size - 1, off)
        b += struct.pack('<3i', *[int(v) for v in np.ravel(L['cb'])])
        l1 = [int(v) for v in np.ravel(L['l1'])]
        b += struct.pack('<9h', *l1)
        b += struct.pack('<H', 0)
        b += struct.pack('<3i', *[int(v) for v in np.ravel(L['lb'])])
        b += struct.pack('<3h', *[int(v) for v in Q['steady']['r'][i]])
        b += struct.pack('<2H', len(w01), len(w2))
        b = align4(b)
        b += struct.pack('<' + 'h' * len(w01), *w01)
        b += struct.pack('<' + 'h' * len(w2), *w2)
    if len(b) > 8192:
        sys.exit(f'{path}: pack is {len(b)} bytes, the loader holds 8192')
    if exact > 8192:
        sys.exit(f'{path}: rings need {exact} entries, nam.c holds 8192')
    return bytes(b), exact


def pack_a2(m, path):
    """one A2 capture as the file nam.c loads. Not the JSON .nam."""
    b, ring = pack_bytes(m, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'wb').write(b)
    return len(b), ring


def emit_a2(k, m, o):
    """A2 nano as its pack bytes (nam.c parses them at load, as it does a pack file); returns (C name, size, ring)"""
    b, ring = pack_bytes(m, m['name'])
    o.append(arr(f'M{k}_PACK', 'uint8_t', list(b), 24))
    return f'M{k}_PACK', len(b), ring


def step(m, x):
    """one sample of the float model (NAM core math); updates m's layer states"""
    v = np.array([x])
    for ly in m['layers']:
        H = len(ly['c'])
        a = ly['W'] @ np.concatenate([v, ly['h']]) + ly['b']
        ly['c'] = sigmoid(a[H:2 * H]) * ly['c'] + sigmoid(a[:H]) * np.tanh(a[2 * H:3 * H])
        ly['h'] = sigmoid(a[3 * H:]) * np.tanh(ly['c'])
        v = ly['h']
    return float(m['head'] @ v + m['head_b'])


def qbits(rows_abs_sum, max_abs, limit=15):
    """largest Q with max|w| 2^Q <= 32767 and every row sum 2^Q < 2^16"""
    q = limit
    while q > 0 and (max_abs * 2 ** q > 32767 or rows_abs_sum * 2 ** q >= 65536):
        q -= 1
    return q


def convert(m):
    sr = m['sr'] or 48000
    for _ in range(int(0.5 * sr)):                         # NAM prewarm: 0.5 s of silence
        step(m, 0.0)
    gain = norm_gain(m)
    out = dict(name=m['name'], H=m['H'], layers=[], gain_db=20 * math.log10(gain), sr=sr)
    for l, ly in enumerate(m['layers']):
        W = ly['W'].copy()
        if l == 0:
            W[:, 0] *= XS                                    # input column sees x / XS
        rows = np.abs(W).sum(1) + np.abs(ly['b'])
        q = qbits(rows.max(), np.abs(W).max())
        Wq = np.round(W * 2 ** q).astype(int)
        bq = np.round(ly['b'] * 2 ** (q + 15)).astype(int)
        hq = np.round(np.clip(ly['h'], -1, 1) * 32767).astype(int)
        cq = np.round(np.clip(ly['c'], -CMAX, CMAX) * 32768).astype(int)
        out['layers'].append(dict(q=q, W=Wq, b=bq, h=hq, c=cq, n_in=ly['n_in'], err=np.abs(Wq / 2 ** q - W).max()))
    hw = m['head'] * gain
    hb = m['head_b'] * gain
    qh = qbits(np.abs(hw).sum() + abs(hb), np.abs(hw).max())
    out.update(head=np.round(hw * 2 ** qh).astype(int), head_b=int(round(hb * 2 ** (qh + 15))), qh=qh)
    return out


def pad(n):
    """head length in nam.c: even (32-bit pair loads) and >= 4 (the dot product's loop runs n / 2 - 1 >= 1 times)"""
    return max(4, n + (n & 1))


def padded(v):
    return np.concatenate([v, np.zeros(pad(len(v)) - len(v), int)])


def pairs(W):
    """rows interleaved by pairs: word (p, j) = W[2p][j], W[2p+1][j] (nam.c computes two rows per MAC instruction)"""
    return np.stack([W[0::2], W[1::2]], axis=2)


def packed(f, xs):
    """table entry k = value(k) | (value(k + 1) - value(k)) << 16, both Q15: one 32-bit load per lookup"""
    v = np.round(f(xs) * 32767).astype(int)
    d = np.append(np.diff(v), 0)
    return [(int(a) & 0xffff) | ((int(b) & 0xffff) << 16) for a, b in zip(v[:-1], d[:-1])]


def arr(name, typ, v, per=12, suffix=''):
    v = [int(x) for x in np.ravel(v)]
    rows = [', '.join(str(x) + suffix for x in v[i:i + per]) for i in range(0, len(v), per)]
    return (f'static const {typ} {name}[{len(v)}] __attribute__((aligned(4))) = {{\n    ' + ',\n    '.join(rows)
            + '\n};')


def main():
    args = sys.argv[1:]
    dst = os.path.join(HERE, 'nam_models.h')
    packs = os.path.join(HERE, 'packs')
    if args[:1] == ['-o']:
        dst, args = args[1], args[2:]
    if args[:1] == ['--packs']:
        packs, args = args[1], args[2:]
    paths = args or sorted(os.path.join(HERE, 'models', f) for f in os.listdir(os.path.join(HERE, 'models'))
                                   if f.endswith('.nam'))
    if not paths:
        sys.exit('no models')
    xs = np.linspace(-8, 8, TAB_N + 1)
    o = ['/* generated by gen_model.py - do not edit */',
         f'#define NAM_XS_SHIFT {int(math.log2(XS))}   /* model input = x_q15 * 2^-15 * {XS:g} */',
         f'#define NAM_CMAX {int(CMAX * 32768)}   /* cell state clamp, Q15 */',
         f'#define NAM_MAX_LAYERS {MAX_LAYERS}', f'#define NAM_MAX_HIDDEN {MAX_HIDDEN}',
         f'#define NAM_NUM_MODELS {len(paths)}']
    lstm_any = any(load(p)['kind'] == 'lstm' for p in paths)
    o.append(f'#define NAM_HAS_LSTM {int(lstm_any)}   /* LSTM code and activation tables only when a model needs them */')
    if lstm_any:
        o += ['/* activations over [-8, 8], 512 segments (step 1/32), Q15; entry = value | delta << 16; Q13 argument */',
              arr('NAM_SIGMOID', 'uint32_t', packed(sigmoid, xs), 6, 'u'),
              arr('NAM_TANH', 'uint32_t', packed(np.tanh, xs), 6, 'u')]
    entries, shapes = [], []
    for k, p in enumerate(paths):
        m = load(p)
        if m['kind'] == 'a2':
            sym, size, ring = emit_a2(k, m, o)
            entries.append(f'    {{ "{m["name"][:15]}", 0, 0, 0, 0, 0, 0, {{ {{ 0 }} }}, {sym}, {size} }},   '
                           f'/* A2 nano, {m["sr"]} Hz, out gain {20 * math.log10(norm_gain(m)):+.1f} dB */')
            print(f'{k}: {m["name"]}  A2 nano WaveNet  ring {ring} entries  gain {20 * math.log10(norm_gain(m)):+.1f} dB  '
                  f'sr {m["sr"]}')
            continue
        c = convert(m)
        shape = (len(c['layers']), c['H'])
        if shape not in shapes:
            shapes.append(shape)
        lay = []
        for l, ly in enumerate(c['layers']):
            o += [f'/* {c["name"]}: layer {l}, Q{ly["q"]}, max weight error {ly["err"]:.2e} */',
                  arr(f'M{k}L{l}_W', 'int16_t', pairs(ly['W'])), arr(f'M{k}L{l}_B', 'int32_t', ly['b']),
                  arr(f'M{k}L{l}_H0', 'int16_t', ly['h']), arr(f'M{k}L{l}_C0', 'int32_t', ly['c'])]
            lay.append(f'{{ {ly["n_in"]}, {ly["q"]}, M{k}L{l}_W, M{k}L{l}_B, M{k}L{l}_H0, M{k}L{l}_C0 }}')
        o.append(arr(f'M{k}_HEAD', 'int16_t', padded(c['head'])))
        entries.append(f'    {{ "{c["name"][:15]}", {len(lay)}, {c["H"]}, {c["qh"]}, {shapes.index(shape)}, {c["head_b"]}, M{k}_HEAD,\n'
                       f'      {{ {", ".join(lay)} }}, 0, 0 }},   /* {c["sr"]} Hz, out gain {c["gain_db"]:+.1f} dB */')
        print(f'{k}: {c["name"]}  {len(lay)}x{c["H"]}  Q{[ly["q"] for ly in c["layers"]]} head Q{c["qh"]}  '
              f'gain {c["gain_db"]:+.1f} dB  sr {c["sr"]}')
    npack = 0
    for k, pth in enumerate(paths):
        m = load(pth)
        if m['kind'] != 'a2':
            print(f'{k}: {m["name"]}  not packed (the device loader is A2 only)')
            continue
        npack += 1
        nbytes, ring = pack_a2(m, os.path.join(packs, f'{npack}.nam'))
        print(f'pack {npack}.nam  {m["name"]}  {nbytes} bytes  ring {ring}')
    o += ['/* (layers, hidden) shapes in use: nam.c compiles one block function per LSTM shape */',
          '#define NAM_SHAPES ' + ' '.join(f'NAM_SHAPE({l}, {h})' for l, h in shapes),
          'static const nam_model_t NAM_MODELS[NAM_NUM_MODELS] = {', *entries, '};']
    open(dst, 'w').write('\n'.join(o) + '\n')


if __name__ == '__main__':
    main()
