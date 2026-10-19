#!/bin/sh
# A2 kernel check: model 0 of the given nam_models.h (an A2 model), blocks x 23 layers through layer_a2, Blackfin
# assembly (simulator) vs the C version (native), every written ring entry and hs word compared; then whole run_a2
# blocks.
#   sh plugins/nam/tools/layer_test.sh models.h [blocks]      # from synth_sdk/, needs the bfin-elf-sim image
set -e
B=harness/build/nam_lt; mkdir -p $B
cp plugins/nam/nam.c plugins/nam/tools/layer_test.c $B/
cp "$1" $B/nam_models.h
F="-DOP1_HOST -DNAM_BAKED -DLAYERS=23 -DBLOCKS=${2:-40} -DRUN_TEST"
docker run --rm -v "$PWD":/work -w /work bfin-elf-sim sh -c "
  bfin-elf-gcc -O2 -mcpu=bf524 -ffreestanding -fno-builtin -nostdlib -fno-common -I. -I$B $F -c $B/layer_test.c -o $B/lt.o &&
  bfin-elf-as -mcpu=bf524 plugins/nam/tools/bench_crt0.S -o $B/crt0.o &&
  bfin-elf-ld -e __start -Ttext=0x1000 $B/crt0.o $B/lt.o \$(bfin-elf-gcc -print-libgcc-file-name) -o $B/lt.elf &&
  bfin-elf-run $B/lt.elf > $B/sim.txt"
cc -O1 -DBENCH_NATIVE $F -I. -I$B $B/layer_test.c -o $B/lt_native
$B/lt_native > $B/native.txt
if cmp -s $B/sim.txt $B/native.txt; then echo "layer_a2 / run_a2: Blackfin bit-exact with C ($(wc -l < $B/sim.txt) lines)"
else echo "MISMATCH:"; diff $B/sim.txt $B/native.txt | head; exit 1; fi
