"""Plugin discovery: every plugins/<folder>/plugin.json is one synth engine or FX.

plugin.json fields (paths relative to the plugin folder):
  name           type name shown on the OP-1: lowercase, <= 15 chars, no clash with built-in types
  kind           "synth" or "fx"
  order          the number stored in saved sounds (knobs[4] = 0x5300 + order), unique per kind, 1..31; never
                 change it once released. Gaps are fine. Taken numbers: plugins/ORDERS.md
  symbol         the op1_engine_t / op1_fx_t object in the sources
  sources        C files; the first one defines `symbol`
  defaults       initial 4 knob positions (0..32767)
  page2_defaults (fx, optional) initial page-2 knob positions
  uses_sample    (synth, optional) engine reads the slot's sampler recording (synth_<slot>.raw loaded at boot)
  screen         (optional) {"svg": file name, must not clash with stock content/display, "symbol": op1_screen_t}
Orders 1..31 per kind (OP1_MAX_ORDER in op1_plugin.h).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, 'plugins')


def load():
    """-> (engines, effects) sorted by order; paths relative to the SDK root (as make_firmware expects)."""
    engs, fxs = [], []
    for d in sorted(os.listdir(DIR)):
        pj = os.path.join(DIR, d, 'plugin.json')
        if not os.path.isfile(pj):
            continue
        p = json.load(open(pj))
        rel = lambda f: os.path.normpath(os.path.join('plugins', d, f))
        e = {k: v for k, v in p.items() if k not in ('kind', 'sources')}
        e['source'], e['extra_sources'] = rel(p['sources'][0]), [rel(s) for s in p['sources'][1:]]
        if p.get('screen'):
            e['screen'] = dict(p['screen'], svg=rel(p['screen']['svg']))
        assert p['kind'] in ('synth', 'fx'), pj
        (engs if p['kind'] == 'synth' else fxs).append(e)
    for lst in (engs, fxs):
        lst.sort(key=lambda e: e['order'])
        orders = [e['order'] for e in lst]
        assert len(set(orders)) == len(orders), f'duplicate plugin order: {orders}'
    return engs, fxs


if __name__ == '__main__':
    for e in sum(load(), []):
        print(e['order'], e['name'], e['source'], e.get('screen', {}).get('svg', '-'))
