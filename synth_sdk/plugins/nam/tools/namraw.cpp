// namraw <model.nam> < in.f32 > out.f32 : NAM core reference, mono float32, no resampling.
#include <cstdio>
#include <filesystem>
#include <vector>
#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
int main(int argc, char** argv) {
  if (argc < 2) return 1;
  auto m = nam::get_dsp(std::filesystem::path(argv[1]));
  if (!m) return 2;
  const int B = 64;
  double sr = m->GetExpectedSampleRate(); m->Reset(sr > 0 ? sr : 48000, B);  // Reset prewarms
  std::vector<float> fi(B); std::vector<NAM_SAMPLE> in(B), out(B);
  size_t n;
  while ((n = fread(fi.data(), 4, B, stdin)) > 0) {
    for (size_t i = 0; i < n; i++) in[i] = fi[i];
    NAM_SAMPLE *ip = in.data(), *op = out.data();
    m->process(&ip, &op, (int)n);
    for (size_t i = 0; i < n; i++) fi[i] = (float)out[i];
    fwrite(fi.data(), 4, n, stdout);
  }
}
