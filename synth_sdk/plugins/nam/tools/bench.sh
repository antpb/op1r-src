#!/bin/sh
# Cycle count of nam.c on the Blackfin simulator, one line per model in the given nam_models.h (default: the plugin's).
#   docker build -t bfin-elf-sim -f plugins/nam/tools/Dockerfile.sim plugins/nam/tools    # once (needs bfin-elf-toolchain)
#   sh plugins/nam/tools/bench.sh [models.h]                                             # from synth_sdk/
set -e
B=harness/build/nam_bench; mkdir -p $B
cp plugins/nam/nam.c $B/nam.c
cp "${1:-plugins/nam/nam_models.h}" $B/nam_models.h
N=$(sed -n 's/^#define NAM_NUM_MODELS \([0-9]*\).*/\1/p' $B/nam_models.h)
docker run --rm -v "$PWD":/work -w /work bfin-elf-sim sh -c "
  bfin-elf-gcc -O2 -mcpu=bf524 -ffreestanding -fno-builtin -nostdlib -fno-common -Wall -I. -DOP1_HOST -DNAM_BAKED \
    -DNAM_NUM_MODELS_FROM_HEADER=$N -c $B/nam.c -o $B/nam.o &&
  bfin-elf-gcc -O2 -mcpu=bf524 -ffreestanding -fno-builtin -nostdlib -fno-common -Wall -I. -DOP1_HOST -DNAM_BAKED \
    -DNAM_NUM_MODELS_FROM_HEADER=$N -c plugins/nam/tools/bench.c -o $B/bench.o &&
  bfin-elf-as -mcpu=bf524 plugins/nam/tools/bench_crt0.S -o $B/crt0.o &&
  bfin-elf-ld -e __start -Ttext=0x1000 $B/crt0.o $B/bench.o $B/nam.o \$(bfin-elf-gcc -print-libgcc-file-name) -o $B/bench.elf &&
  bfin-elf-run $B/bench.elf > $B/sim.txt"
cat $B/sim.txt
cc -O2 -DOP1_HOST -DNAM_BAKED -DBENCH_NATIVE -DNAM_NUM_MODELS_FROM_HEADER=$N -I. $B/nam.c plugins/nam/tools/bench.c -o $B/bench_native
$B/bench_native > $B/native.txt
cc -O2 -DOP1_HOST -DNAM_BAKED -DBENCH_NATIVE -DNAM_NO_REST -DNAM_NUM_MODELS_FROM_HEADER=$N -I. $B/nam.c plugins/nam/tools/bench.c -o $B/bench_awake
$B/bench_awake > $B/awake.txt
if [ "$(sed 's/.*hash //' $B/sim.txt)" = "$(sed 's/.*hash //' $B/native.txt)" ]; then echo "Blackfin output bit-exact with the C reference"
else echo "MISMATCH: Blackfin vs C reference"; cat $B/native.txt; exit 1; fi
if [ "$(sed 's/.*hash //' $B/native.txt)" = "$(sed 's/.*hash //' $B/awake.txt)" ]; then echo "sleeping through silence is bit-exact"
else echo "MISMATCH: sleeping vs always running"; cat $B/awake.txt; exit 1; fi
