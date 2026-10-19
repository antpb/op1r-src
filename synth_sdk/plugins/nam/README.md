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
way. The LSTM path (1.5 KB plus 4 KB of tables) is compiled only for that bench, when a baked model needs it.
The firmware image has no room for those tables; see below.

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
`render()`. The four baked models are all A2 nano and land on the same count:

| | cycles / sample | 400 MHz |
|---|---|---|
| playing | 1 884 | **20.7 %** |
| silent input, model asleep | 54 | 0.6 % |

The first kernel was 2 877 cycles (31.7 %). A per-layer profile of the current one, same simulator:

| | cycles / frame |
|---|---|
| one K = 6 layer (21 of the 23) | 70 |
| one K = 15 layer (2 of the 23) | 115 |
| all 23 layers | 1 717 |
| whole model, including input, head and output | 1 905 |

The 1 905 is one profiled block of the model, still inside the fade-in. The 1 884 is `render()` averaged
over steady blocks, after the fade, and it includes the DC blocker, tone and level.

Five changes make up the drop from 31.7 %, and they are the balance to keep. Each one is bit-exact with
`a2model.run_int` (`compare.py` with no `--ref`).

1. **The layer loop** (`layer_a2`, Blackfin). One hardware loop per layer, 5K + 38 instructions a frame.
   Biases are the first words of the stream and seed the accumulators. The head sum is updated in the same
   instructions as the stores. The old loop spent about 70 instructions a frame on bias loads, a head-sum
   read-modify-write and load-use stalls, against about 30 for the taps.
2. **Exact-length rings.** (K − 1)·D + 64 entries, 7 803 in total, 61 KB. Power-of-two rings were 104 KB, and
   every block streamed the padding through the cache. The DAG circular buffers take any length, so the
   rounding was pure traffic. A model change now fills in about 6 ms.
3. **16-bit multiply and saturate** on the input, the head and the output (`mul16`, `sat16` in `nam.c`). gcc
   was emitting a 32×32 multiply, which is several cycles on this core, and a compare plus two branches for
   each saturation. The simulator counts every instruction as one cycle and models no stalls, so it shows this
   fix as 2 024 → 1 884 (22.3 % → 20.7 %). On the device the 32×32 multiply is the expensive one, and the
   saving is larger than that gap.
4. **Sleep.** After 6 346 zero samples (Σ (K − 1)·D + 15, about 144 ms) every history is the integer steady
   state and the head output is the constant `yrest`. Later all-zero blocks skip the model and run the DC
   blocker, tone and level on that constant. `tools/bench.sh` checks the hash against a build that never
   sleeps (`-DNAM_NO_REST`). The gate multiplies the model input, so a closed gate is exact zeros and a pause
   between phrases costs the 0.6 %. The gate off leaves the hiss filter in the path, the input is never exact
   zeros, and the model never sleeps.
5. **The slot's synth is not rendered** while `nam input` is the synth FX. Its output was discarded. Each
   block the effect stores `op1_audio_blocks` in `nam_synth_stamp`, and `synth_render` (`dispatcher.c`) writes
   silence while that stamp is this block or the previous one. The one-block window is there because the voice
   render and the FX render can run in either order inside one block. The 1 884 above is the effect alone;
   the voices used to run on top of it. This is the part that was making tape recording hitch. It covers the
   synth FX position only. On the master FX the dry synth is part of the mix, and the voices still run.

**20.7 % while a note is sounding is still heavy.** op1r's plugins aim at about 8 %. The layer loop is close
to the multiply count (2 cycles a tap for output 2, 3 for outputs 0 and 1), so a faster kernel is not where
the next saving is. A smaller capture is. A standard WaveNet is about 15 000 multiplies a sample and does not
fit. The simulator has no SDRAM waits, the histories live in SDRAM, and the real load is higher than the
table. **Read the T1 `nam` line on the device before relying on it**, including while recording to tape. If
the audio callback goes over 100 % the OP-1 can lock up until it is switched off. The effect runs only while
it is selected.

## Files and checks

| file | |
|---|---|
| `plugin.json`, `naminput.c`, `naminput.svg` | the listed effect: screen, and the live-input entry |
| `nam.c` | the amp, the pack loader, the bench entry `FX_nam` |
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

## For the next firmware change

**The code image is full.** Custom code is linked at `0x01E00000` and has to end before `0x01E40000`
(`CODE_LIMIT` in `make_firmware.py`, `__image_end` in `link.ld`, docs/30). Another project's module starts
at that address, and the pack step asserts. With this effect in the tree the loaded image (`.text` +
`.rodata` + `.data`) ends **324 bytes** below the limit. Before the faster layer loop there was about 2 KB
free; the assembly and the input/head path took roughly 1.7 KB of it. `.bss` is a different region, at
`0x01EC0000`, and still has room under `0x01EF0000`. The tight one is the loaded image.

That is why the weights are a 5.5 KB pack on disk, parsed into the FX scratch (`OP1_FX_SCRATCH_BYTES`,
176 460 bytes per position), and why the LSTM activation tables exist only in the host bench. A new plugin,
a table, or an unrolled copy of this loop will fail the firmware build. Anything big has to replace code
already in the image, or live in scratch or on disk.

Things that took a while to learn, and that a later change can break quietly:

- **gcc widens 16-bit multiplies.** On the per-sample path, write the multiply and the saturate as the
  Blackfin 16-bit ops (`mul16`, `sat16`). A C multiply of two `int16_t` values becomes a 32×32 multiply,
  several cycles, and the simulator still counts it as one. The same applies to a saturate written as two
  compares.
- **Inline asm.** `P5` is the PIC register; clobbering it fails the compile. `L0`, `L1` and `L2` have to be
  0 on the way out, or later circular addressing in the firmware fires on a stale length. The accumulator
  extract is `A0` into an even register and `A1` into an odd one.
- **The simulator is a floor.** `bfin-elf-run` counts one cycle per instruction and models no memory waits.
  The histories are in SDRAM. The number to trust is the T1 `nam` line while the OP-1 is recording.
- **Sleep is exact, and only on exact zeros.** It is safe because the integer model really does sit on
  `yrest` after one receptive field of zeros. Shortening that (6 346 samples) or sleeping on "quiet" input
  changes the sound. `tools/bench.sh` is the check: its hash has to match the `-DNAM_NO_REST` build.
- **The three implementations have to stay in lockstep.** `a2model.run_int`, the C `layer_a2`, and the
  assembly. LeakyReLU uses the Blackfin's convergent rounding (ties to even), and both other copies repeat
  it. `tools/layer_test.sh` compares every written history word, including across the ring wrap.
- **The voice skip is a one-block window on purpose.** `nam_synth_stamp` against `op1_audio_blocks`, tested
  with `<= 1`, because the voice and the FX can be ordered either way inside a block. Tightening it to the
  current block makes the synth render on every other block. It is wired for the synth FX position only.

Still open:

- Nothing here has been timed on the OP-1. The 20.7 % and the 0.6 % are simulator floors. Confirm on T1,
  with tape recording the effect output, before treating the recording hitch as fixed.
- While a note is sounding the effect is still about 21 %. The next saving is a smaller model. The kernel
  is already close to the multiply count, and there is no code space to grow it.
- Captures are 48 kHz and play at 44.1 kHz, so the response sits about 8 % low. A resampler does not fit
  in the image or the budget.
- With the gate off, the model never sleeps.
- On the master FX, the slot's voices still render.
