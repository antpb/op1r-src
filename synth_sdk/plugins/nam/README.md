# nam: Neural Amp Modeler on the OP-1

An amp FX that runs real [Neural Amp Modeler](https://github.com/sdatkinson/NeuralAmpModelerCore) captures on
the OP-1's Blackfin. It has no FPU, so everything runs in fixed point. This folder is that effect: `plugin.json`
registers `nam input` (`naminput.c` is the screen and the live-input entry, `nam.c` is the amp).

The capture is not built into the firmware. Put `1.nam`, `2.nam`, up to `16.nam` in a `nam` folder at the top of
the disk the OP-1 shows when it is mounted, next to `album`, `synth`, `drum`, and `tape`. Leave disk mode.
That cleanup used to delete any other root folder. It now leaves `nam` in place, copies the packs, and the
effect plays the copy. The number on the screen is the file. The blue encoder's press steps to the next one
that is there, drops the model it had loaded, and reads the new one. A missing number is skipped. One model
is in memory at a time.

Those files are not the JSON `.nam` a capture comes as. The OP-1 cannot run that conversion. `gen_model.py`
writes the fixed-point pack:

```sh
python3 plugins/nam/gen_model.py --packs /Volumes/OP-1/nam a.nam b.nam   # -> 1.nam, 2.nam, in that order
python3 plugins/nam/gen_model.py                                         # models/*.nam -> plugins/nam/packs/
```

Copy `packs/1.nam` (and `2.nam`, …) into the OP-1's `nam` folder if you did not write them there directly.

## Knobs

| knob | function |
|---|---|
| blue | GAIN: input drive, −18 … +18 dB (0 dB in the middle) |
| blue press | next profile in the `nam` folder (`1.nam`, `2.nam`, …). The screen shows the number |
| green | TONE: 12 dB/oct low-pass after the amp, 1 kHz … open (fully up = off) |
| white | GATE: opens on the input peak, off at 0, −90 … −30 dBFS (default −50). Holds while the average is within 18 dB, and not below about −70 |
| red | LEVEL: output, unity at 70 %, +6 dB at the top |

**Playing a guitar through it.** `nam input` is the only listed effect. It plays the line input instead of the
slot's synth, but it stays silent until two things are true: a cable is in the input jack (the source is line,
not the built-in mic), and the mic/input key has external audio on. Nothing plugged in leaves the codec on the
internal mic, and running that through the amp feeds the speaker straight back into the mic. With the mic key
off, or with the mic still selected, the effect outputs silence and does not run the model. Press the mic/input
key once the jack shows line; the guitar then runs continuously through gate → amp → tone into that synth's bus,
with no key to hold, and records to tape like a synth. That same key also mixes the dry input around the FX, so
you hear the dry guitar as well as the amp. Synth engines can't do this, because the firmware only renders a
voice while a key is held.

The slot's synth is never heard while `nam input` replaces its bus, so its voices are not rendered either: each
block the effect stamps the audio block counter, and the voice render (`dispatcher.c`) outputs silence while that
stamp is from the current or the previous block. Keys played on the slot cost nothing; switching to another effect
brings the synth back on the next block.

How the input is read: the audio callback is called with (output block, input block, …), each 64 frames of
(L, R) 32-bit words. The firmware's hardware test modes show this: `0x0191A7A8` copies input to output, and
`0x0191A880` writes tones to the output. op1r's callback wrapper (`screens/sysmon.c`) copies the input block at the
start of each callback into `op1_audio_in` (`op1_plugin.h`); the FX runs inside the same callback and reads it.
Samples are the top 16 bits. The louder of L / R is used (smoothed, switching at 2×), so a mono cable works on
either side. The level compare keeps the top 16 bits of the 32-bit word; an earlier compare threw away another
6 bits and went deaf near −54 dBFS, so a quiet guitar on the ring never took the channel. The T1 monitor shows
`nam 21 pk30 run100 a… s…`: the effect's share of the audio time now and its peak (%), the share of blocks in
which the model ran (the rest slept, see CPU), then the input block address (alternates `…370` / `…570`) and blocks
read. A first version followed the RX DMA position itself and read inconsistent blocks.

**Why a gate.** An amp capture adds a lot of gain to small signals: the Twin turns −70 dBFS of input hiss into
about −38 dBFS of static, and float NAM core does the same. The OP-1 input hisses even with nothing plugged in.
The gate scales the model's input, so once it is closed the model gets exact zeros and the output settles to
digital silence. It opens in about 1.5 ms and closes with a ~70 ms fade.

The open detector is a peak follower, not the ~6 ms mean. A guitar's mean sits about 14 dB under its peaks, and
the mean almost ignores a pick (a 1 ms attack only reaches about a sixth of its height in 6 ms), so a note turned
down toward half volume never crossed −50 dBFS on the average and the gate stayed shut. The follower takes half
the remaining step each sample and releases in ~45 ms, so it sits on the attack and does not ripple at the
fundamental. On hiss it rests near −59 dBFS (the spikes themselves are near −58), under the −50 default, so
silence stays shut. It then holds while the 6 ms mean is within 18 dB of the threshold — the mean is already
~14 dB down, so 12 dB of hysteresis never covered the tail — and for ~35 ms after both let go. The close level
is not allowed below about −70 dBFS: the hiss mean is near −73, and a lower close latches the gate open.

**Input filters, live input only.** Two one-pole high-passes at ~48 Hz run before the gate and the model. They
remove DC, subsonic rumble and part of the 60 Hz hum, which the capture would otherwise amplify (low E, 82 Hz,
is about −2.5 dB; 20 Hz is about −16 dB). A DC offset also used to sit in the level detector as if it were
signal. Above ~8.8 kHz a one-pole split is turned down to −12 dB while that band is only hiss, and a pick opens
it in under a millisecond. The steady hiss in that band drops about 11 dB; the body of the note is not on that
path, and a loud note keeps its top end (about −3 dB there). Turning the gate off bypasses the gate and the
hiss split. The rumble filter stays.

Drive is constant for the block. A per-sample chase of the knob dithers the model input, and the capture turns
that into sputtering on a ringing chord and on the noise floor. The head output of a hot capture swings past
full scale; the DC blocker removes that offset and the clip is after it. Once the gate has closed, the input
peak has to fall back under the threshold before it will open again, so the tail of a chord cannot re-trigger it.

The output is DC-blocked and loudness-normalised to −18 dB from the capture's metadata. A model change mutes for
about 6 ms while the histories are filled with the model's settled silence state, then fades in.

## Which models run

| architecture | weights | multiplies / sample | on the OP-1 |
|---|---|---|---|
| WaveNet standard (most captures) | 13 802 | ~15 000 | no: more than the whole CPU |
| **A2 nano** (3 channels; the small submodel of an A2 SlimmableContainer) | 1 871 | ~1 730 | yes |
| LSTM up to 2 × 8 | ≤ 1 000 | ≤ 300 | yes |

`gen_model.py` reads `models/*.nam` and takes the A2 nano submodel out of a SlimmableContainer by itself. Captures
are made at 48 kHz and run at 44.1 kHz here (there is no room for a resampler), so the response sits about 8 %
lower in frequency.

**Space.** The weights are not in the firmware. One A2 pack is about 5.5 KB and is read into the FX scratch,
after the histories; the loader turns it into the model and the weight streams the kernel reads, also in the
scratch. Up to 16 packs can sit in the `nam` folder; only the selected one is loaded. Switching discards it and
reads the next. The host bench (`-DNAM_BAKED`) has the same pack bytes in `nam_models.h` and parses them the same
way. The LSTM path (1.5 KB plus 4 KB of tables) is only compiled there, when a baked model needs it.

## Fixed point (A2)

The format is in `a2model.py`. Activations are int16 with a power-of-two scale per layer and per tensor
(residual stream, layer output, head sum). The scales are measured on a calibration signal at +12 dB drive, and
the stores saturate above that. Every accumulator is proven to fit in int32 from the weights' row norms, and the
rounding offsets are folded into the biases. Near silence the fixed point adds a floor of about −46 dBFS (float:
−47 dBFS for −80 dBFS of hiss, fixed: −43 dBFS); the gate removes it in pauses. Each history holds exactly its
lookback plus one block, (K − 1)·D + 64 entries: 7 803 entries × 8 bytes = 61 KB for an A2 nano (power-of-two
rings were 104 KB). They live in the FX position's scratch buffer, and so do the block buffers: the audio
interrupt's stack is small. Each layer runs over the whole 64-sample block, which keeps its ring region and
weights cache-hot. Packs still carry power-of-two ring sizes for older loaders; nam.c ignores them.

Three implementations compute identical bits:

- `a2model.run_int`: Python, vectorised. The formats were designed on it.
- `layer_a2` in C: the host build and the harness.
- `layer_a2` in Blackfin assembly: one loop per layer, 5K + 38 instructions a frame. Rings, weight streams and
  the next layer's ring use circular DAG registers, so a ring can have any length. Outputs 0 and 1 take 3
  dual-MAC cycles per tap, output 2 takes 2; biases seed the accumulators from the stream, and the head-sum
  update shares its instructions with the loads and stores around it. LeakyReLU is a vector fractional multiply
  plus MAX, and its rounding is the Blackfin's convergent rounding (ties to even), which the C and Python
  versions copy.

## Accuracy against NAM core (float)

From `compare.py`. The same input goes to NAM core (`tools/namraw.cpp`, the extracted 3-channel submodel) and to
nam.c, with the plugin's gain, DC blocker and clip applied to the reference.

| signal | 0 dB drive | +6 dB | +12 dB |
|---|---|---|---|
| saw chords | 28.3 dB | 32.3 dB | 38.3 dB |
| plucks | 31.3 | 35.0 | 39.9 |
| sine sweep | 46.7 | 48.1 | 48.8 |
| guitar DI | 31.4 | 34.5 | 38.3 |

These are SNRs for the Twin. It is the most sensitive of the five A2 slim captures tried. The soy_milk captures
reach 42–50 dB, and the 12AX7 4vol one 34–37 dB. For scale, a NAM capture is typically ~1 % ESR (−20 dB) away
from the real amp, so these errors are 10–30 dB below the model's own error. Silence stays silent: the integer
model sits exactly at its steady state, so there is no idle hiss.

## CPU

From `tools/bench.sh`, the gdb Blackfin simulator (`bfin-elf-run`), reading the CYCLES register around
`render()`:

| | cycles / sample | 400 MHz |
|---|---|---|
| A2 nano, playing | 1 884 | **20.7 %** |
| A2 nano, silent input (model asleep) | 54 | 0.6 % |
| LSTM 1×3 (for comparison) | 605 | 6.6 % |

Per frame, a K = 6 layer takes 70 cycles and a K = 15 layer 115 (1 717 for all 23); the input, head and output
stages take the rest. The first kernel took 2 877 cycles (31.7 %): about 70 instructions per layer and frame
went to bias loads, the head-sum read-modify-write and load-use stalls.

**Sleep.** Once the model has taken in zeros for its whole receptive field (Σ (K − 1)·D + 15 = 6 346 samples,
~144 ms), every history sits at the integer steady state and the head output is a constant. Blocks of exact zeros
after that skip the model and only run the DC blocker, tone and level on that constant: bit-identical output
(`tools/bench.sh` checks it against a build that never sleeps, `-DNAM_NO_REST`). With the gate on, the model's
input is exact zeros once the gate has closed, so pauses between phrases cost almost nothing.

**This is still heavy.** op1r's plugins aim at ~8 %. The simulator also has no SDRAM wait states, and the
histories live in SDRAM, so the real load is probably higher. **Check it on the T1 resource monitor (the `nam`
line) before relying on it.** If the total goes over 100 %, the OP-1 can lock up until it is switched off. The
FX only runs while it is selected.

## Files and checks

| file | |
|---|---|
| `nam.c` | the plugin |
| `gen_model.py` | `models/*.nam` → `nam_models.h` (host bench) and `packs/N.nam` (what the OP-1 loads) |
| `a2model.py` | A2: float reference, calibration, conversion, integer reference |
| `compare.py [--ref <namraw>] [--di wav]` | accuracy against NAM core, plus bit-exactness against `run_int` (only that without `--ref`) |
| `tools/namraw.cpp` | NAM core renderer, raw float32 stdin → stdout |
| `tools/bench.sh [models.h]` | simulator cycles (playing and asleep), bit-exactness hashes against a native build and one that never sleeps |
| `tools/layer_test.sh models.h` | asm vs C, every word, 40 blocks × 23 layers (every ring wraps) |
| `tools/Dockerfile.sim` | `bfin-elf-sim` image: the toolchain image plus gdb 13's Blackfin simulator |

```sh
# NAM core renderer (needs an Eigen with Eigen::lastN, e.g. NeuralAmpModelerCore's Dependencies/eigen)
N=path/to/NeuralAmpModelerCore
clang++ -std=c++20 -O3 -include plugins/nam/tools/eigen_lastn_shim.h -I$N -I$N/Dependencies/eigen \
  -I$N/Dependencies/nlohmann plugins/nam/tools/namraw.cpp $N/NAM/*.cpp $N/NAM/*/*.cpp -o namraw
docker build -t bfin-elf-sim -f plugins/nam/tools/Dockerfile.sim plugins/nam/tools    # once
```

(The shim maps `Eigen::placeholders::lastN` onto older Eigens; drop it with a current Eigen.)

## Not done

- 48 kHz captures run at 44.1 kHz.
- CPU: see above. The pack is read from the disk into scratch, so the weights are no longer in the code image.
