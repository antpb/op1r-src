"""NAM "A2 nano" WaveNet (3 channels) for nam.c: float reference, calibration, fixed-point conversion, and an integer
reference that computes exactly what nam.c computes (vectorised over time: a WaveNet has no recurrence).

A2 shape (NAM core wavenet/a2_fast.h): one layer array, 23 layers, kernel 6 (15 for layers 14, 15), dilations
1 3 7 17 41 101 239 (x2), 1 13, 1 3 7 17 41 101 239; bottleneck = channels; LeakyReLU(0.01); 1x1 residual per
layer; head = 16-tap conv over the sum of all layer activations, times head_scale (the last weight). Per sample:
    r = rech x;  for each layer: z = lrelu(cb + sum_k W_k r[t - (K-1-k) D] + mix x);  hs += z;  r += L1 z + lb
    y = head_scale (hb + sum_k Hw_k hs[t - 15 + k])
Fixed point (all exponents are powers of two, measured on a calibration signal at +12 dB drive):
    x      int16, value = q 2^(XE - 15)                (XE = 3: the drive knob's +18 dB headroom, as for LSTMs)
    r_0    = sat((rech x) >> s0)
    r_l    int16, value = q 2^(er_l - 15)              residual stream entering layer l (er_23: unused)
    z_l    int16, value = q 2^(ez_l - 15)
    conv   acc = cb + sum W q_r + mix q_x  ->  z = acc >> (Qc + ez - er)      (mix weights in conv units)
    lrelu  max(z, rnd(z 328 / 2^15))                  (0.01001; the Blackfin's fractional multiply: round to
                                                       nearest, ties to even = its default convergent rounding)
    hs     int32, value = q 2^(ez_min - 15); per layer hs += z << (ez - ez_min)
    1x1    acc = (r << (er - ez + Ql)) + sum L1 z + lb  ->  r' = acc >> (Ql + er' - ez)
    head   hq = hs >> (eh - ez_min) (int16 ring); acc = hb + sum Hw hq -> y (Q15) = acc >> (Qh - eh)
Every accumulator is proven to stay inside int32 from the weights (row L1 norms), all shifts are >= 1 and their
rounding (+half) is folded into the biases. Stores saturate to int16.
"""
import json
import numpy as np

KS = [6] * 14 + [15, 15] + [6] * 7
BLOCK = 64                                                     # nam.c runs each layer over a 64-sample block
DS = [1, 3, 7, 17, 41, 101, 239] * 2 + [1, 13] + [1, 3, 7, 17, 41, 101, 239]
C = 3
XE = 3
LEAK_Q15 = 328
HEADROOM = 1.0                                         # over the calibration peak (stores saturate)
CAL_DRIVE = 4                                         # calibrate at +12 dB drive: best SNR over 0 .. +18 dB


def is_a2_nano(m):
    if m.get('architecture') != 'WaveNet':
        return False
    c = m.get('config', {})
    ls = c.get('layers') or []
    if len(ls) != 1 or c.get('head') is not None:
        return False
    l = ls[0]
    acts = l.get('activation')
    acts = acts if isinstance(acts, list) else [acts] * 23
    return (l.get('channels') == C and l.get('bottleneck', C) == C and l.get('kernel_sizes') == KS and
            l.get('dilations') == DS and l.get('input_size') == 1 and l.get('condition_size') == 1 and
            (l.get('head') or {}).get('kernel_size') == 16 and not (l.get('head1x1') or {}).get('active') and
            (l.get('layer1x1') or {}).get('active', True) and
            all(isinstance(a, dict) and a.get('type') == 'LeakyReLU' and abs(a.get('negative_slope', 0) - 0.01) < 1e-9
                for a in acts) and
            all(g == 'none' for g in l.get('gating_mode', ['none'])))


def pick(d):
    """the A2 nano model inside a .nam: the file itself, or the smallest submodel of a SlimmableContainer"""
    if d.get('architecture') == 'SlimmableContainer':
        for sm in d['config']['submodels']:
            m = sm.get('model', sm)
            if is_a2_nano(m):
                return dict(m, sample_rate=m.get('sample_rate') or d.get('sample_rate'),
                            metadata=m.get('metadata') or d.get('metadata'))
        return None
    return d if is_a2_nano(d) else None


def parse(m):
    w = iter(np.asarray(m['weights'], dtype=np.float64))
    t = lambda n: np.array([next(w) for _ in range(n)])
    P = {'rech': t(C), 'layers': []}
    for K, D in zip(KS, DS):
        conv = t(C * C * K).reshape(C, C, K)                  # [out][in][tap]
        P['layers'].append(dict(K=K, D=D, conv=conv, cb=t(C), mix=t(C), l1=t(C * C).reshape(C, C), lb=t(C)))
    P['head'] = t(C * 16).reshape(C, 16)                      # [in][tap]
    P['hb'] = t(1)[0]
    P['hs'] = t(1)[0]                                         # head_scale: the last weight (overrides the JSON)
    assert next(w, None) is None, 'A2 weight count mismatch'
    return P


def run_float(P, x, trace=None):
    """float model over a whole signal (x starts after silence: NAM's prewarm), like NAM core"""
    W = 8000
    x = np.concatenate([np.zeros(W), np.asarray(x, np.float64)])
    n = len(x)
    r = np.outer(x, P['rech'])
    hs = np.zeros((n, C))
    for L in P['layers']:
        K, D = L['K'], L['D']
        pad = np.vstack([np.repeat(r[:1], (K - 1) * D, 0), r])
        z = np.tile(L['cb'], (n, 1)) + np.outer(x, L['mix'])
        for k in range(K):
            z += pad[k * D:k * D + n] @ L['conv'][:, :, k].T
        z = np.where(z >= 0, z, 0.01 * z)
        if trace is not None:
            trace['r'].append(np.abs(r).max())
            trace['z'].append(np.abs(z).max())
        hs += z
        r = r + z @ L['l1'].T + L['lb']
    if trace is not None:
        trace['h'] = np.abs(hs).max()
    pad = np.vstack([np.repeat(hs[:1], 15, 0), hs])
    y = np.full(n, P['hb'])
    for k in range(16):
        y += pad[k:k + n] @ P['head'][:, k]
    return (y * P['hs'])[W:]


def calibration_signal(sr=48000, seed=5):
    """plucked saws / chords, noise bursts and a sweep, peak 1.0, then at the drive knob's maximum (x 2^XE)"""
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(40):
        n = int(sr * rng.uniform(0.2, 0.8))
        t = np.arange(n) / sr
        f = 55 * 2 ** (rng.integers(0, 48, rng.integers(1, 4)) / 12)
        x = sum(2 * ((t * fi + rng.random()) % 1) - 1 for fi in f) / len(f)
        parts.append(x * np.exp(-t * rng.uniform(2, 10)) * 10 ** (rng.uniform(-20, 0) / 20))
        parts.append(rng.standard_normal(int(sr * 0.05)) * 10 ** (rng.uniform(-30, -6) / 20))
    t = np.arange(2 * sr) / sr
    parts.append(np.sin(2 * np.pi * np.cumsum(40 * (250 ** (t / t[-1]))) / sr))
    x = np.concatenate(parts)
    return np.clip(x / np.abs(x).max(), -1, 1)


def qbits(max_abs, row_l1_x32768, limit=15):
    """largest Q with max|w| 2^Q <= 32767 and row_l1 2^Q < 2^30 (row_l1 already in units of the 2^15 input)"""
    q = limit
    while q > 0 and (max_abs * 2 ** q > 32767 or row_l1_x32768 * 2 ** q >= 2 ** 30):
        q -= 1
    return q


def ceil_log2(v):
    return int(np.ceil(np.log2(max(v, 1e-6))))


def convert(m, gain):
    """fixed-point model; gain = output normalisation (folded into the head)"""
    P = parse(m)
    tr = {'r': [], 'z': []}
    run_float(P, calibration_signal() * CAL_DRIVE, tr)
    er = [ceil_log2(v * HEADROOM) for v in tr['r']] + [0]
    ez = [ceil_log2(v * HEADROOM) for v in tr['z']]
    ez_min = min(ez)
    eh = max(ceil_log2(tr['h'] * HEADROOM), ez_min + 1)
    er[0] = max(er[0], XE - 14)                               # r_0 = rech x: keep it representable
    Q = {'layers': [], 'ez_min': ez_min, 'eh': eh}
    for l, L in enumerate(P['layers']):
        K = L['K']
        e_r, e_z, e_r2 = er[l], ez[l], er[l + 1] if l + 1 < 23 else er[l]
        # conv: input r (2^e_r), x (2^XE). Weight in "conv units": W 2^Qc, mix 2^(Qc + XE - e_r)
        mixs = 2.0 ** (XE - e_r)
        row = np.abs(L['conv']).sum(axis=(1, 2)) + np.abs(L['mix']) * mixs
        qc = qbits(max(np.abs(L['conv']).max(), np.abs(L['mix']).max() * mixs),
                   (row.max() + np.abs(L['cb']).max() * 2.0 ** -e_r) * 32768)
        sz = qc + e_z - e_r
        while sz < 1:                                         # (never for real models: z is not tiny vs r)
            e_z += 1; sz += 1
        # 1x1: acc = (r << s_in) + L1 z + lb, units 2^(e_z - Ql - 15)
        ql = 15
        while ql > 0:
            s_in = e_r - e_z + ql
            bound = 32767 * 2.0 ** s_in + np.abs(L['l1']).sum(1).max() * 2 ** ql * 32768 + \
                np.abs(L['lb']).max() * 2.0 ** (ql + 15 - e_z)
            if 0 <= s_in <= 14 and np.abs(L['l1']).max() * 2 ** ql <= 32767 and bound < 2 ** 30 and ql + e_r2 - e_z >= 1:
                break
            ql -= 1
        s_in, sr_ = e_r - e_z + ql, ql + e_r2 - e_z
        assert 0 <= s_in <= 14 and sr_ >= 1, (l, s_in, sr_)    # 2^s_in is a 16-bit MAC operand in nam.c
        cu = 2.0 ** (e_r - qc - 15)                           # value of one conv accumulator LSB
        lu = 2.0 ** (e_z - ql - 15)                           # value of one 1x1 accumulator LSB
        Q['layers'].append(dict(
            K=K, D=L['D'], qc=qc, ql=ql, er=e_r, ez=e_z, er2=e_r2, sz=sz, sr=sr_, s_in=s_in, sh=e_z - ez_min,
            W=np.round(L['conv'].transpose(2, 1, 0) * 2 ** qc).astype(np.int64),   # [tap][in][out]
            mix=np.round(L['mix'] * mixs * 2 ** qc).astype(np.int64),
            cb=np.round(L['cb'] / cu).astype(np.int64) + (1 << (sz - 1)),
            l1=np.round(L['l1'] * 2 ** ql).astype(np.int64),                       # [out][in]
            lb=np.round(L['lb'] / lu).astype(np.int64) + (1 << (sr_ - 1))))
    s0 = 15                                                   # r_0 = sat((rech x_q) >> s0)
    while np.abs(P['rech']).max() * 2.0 ** (XE - er[0] + s0) > 32767:
        s0 -= 1
    assert s0 >= 1, 'rechannel shift'
    Q['rech'] = np.round(P['rech'] * 2.0 ** (XE - er[0] + s0)).astype(np.int64)
    Q['s0'] = s0
    Q['er0'] = er[0]
    hw = P['head'] * P['hs'] * gain                           # [in][tap]
    hb = P['hb'] * P['hs'] * gain
    qh = 15
    while qh > 0 and (np.abs(hw).max() * 2 ** qh > 32767 or
                      (np.abs(hw).sum() * 2 ** qh * 32768 + abs(hb) * 2.0 ** (qh + 15 - eh)) >= 2 ** 30 or qh - eh < 1):
        qh -= 1
    assert qh - eh >= 1, 'head shift'
    Q['qh'] = qh
    Q['head'] = np.round(hw.T * 2 ** qh).astype(np.int64)                      # [tap][in]
    Q['hb'] = int(round(hb * 2.0 ** (qh + 15 - eh))) + (1 << (qh - eh - 1))
    Q['steady'] = steady(Q)
    return Q


def ring_size(K, D):
    """history ring entries in nam.c: the lookback plus one block (written before the block's taps)"""
    return (K - 1) * D + BLOCK


def _sat(v):
    return np.clip(v, -32768, 32767)


def _layer_int(L, r_hist_fn, xq, rq):
    """one layer over a whole signal: r_hist_fn(lag) -> r delayed by lag samples; returns (z, r')"""
    acc = np.tile(L['cb'], (len(xq), 1)) + np.outer(xq, L['mix'])
    K, D = L['K'], L['D']
    for k in range(K):
        acc += r_hist_fn((K - 1 - k) * D) @ L['W'][k]
    z = _sat(acc >> L['sz'])
    v = z * LEAK_Q15
    q = (v + 16384) >> 15
    q = q - (((v & 0x7fff) == 0x4000) & ((q & 1) == 1))     # tie: to even
    z = np.maximum(z, q)
    acc2 = (rq << L['s_in']) + z @ L['l1'].T + L['lb']
    return z, _sat(acc2 >> L['sr'])


def steady(Q):
    """integer steady state for silence (NAM's prewarm): r entering each layer, the head ring value"""
    rq = np.zeros((1, C), np.int64)
    rs, hs = [], np.zeros((1, C), np.int64)
    for L in Q['layers']:
        rs.append(rq[0].copy())
        z, rq = _layer_int(L, lambda lag, r=rq: r, np.zeros(1, np.int64), rq)
        hs += z << L['sh']
    hq = _sat(hs >> (Q['eh'] - Q['ez_min']))
    return {'r': rs, 'h': hq[0]}


def run_int(Q, xq):
    """exactly nam.c's model output (Q15, before DC blocker / tone / level) for int16 input xq, from the steady state"""
    xq = np.asarray(xq, np.int64)
    n = len(xq)
    rq = _sat((np.outer(xq, Q['rech'])) >> Q['s0'])
    hs = np.zeros((n, C), np.int64)
    for l, L in enumerate(Q['layers']):
        pre = np.tile(Q['steady']['r'][l], (L['K'] - 1) * L['D']).reshape(-1, C) if L['K'] > 1 else np.zeros((0, C))
        full = np.vstack([pre.astype(np.int64), rq])
        off = len(pre)
        z, rq = _layer_int(L, lambda lag, f=full, o=off: f[o - lag:o - lag + n], xq, rq)
        hs += z << L['sh']
    hq = _sat(hs >> (Q['eh'] - Q['ez_min']))
    full = np.vstack([np.tile(Q['steady']['h'], 15).reshape(-1, C), hq])
    acc = np.full(n, Q['hb'], np.int64)
    for k in range(16):
        acc += full[k:k + n] @ Q['head'][k]
    return acc >> (Q['qh'] - Q['eh'])


def load_json(path):
    return json.load(open(path))
