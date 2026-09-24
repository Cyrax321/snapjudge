# snapjudge CUDA fast path — port of tl_kernels.py + fast.py.
#
# Only built when SNAPJUDGE_CUDA=ON and a CUDA toolkit is present. The macOS
# dev box has no NVIDIA GPU, so these kernels are compile- and review-verified
# here; runtime parity vs the bf16 autocast stock path is exercised by
# `bench/parity_fast.py` on a CUDA machine (see README).

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(snapjudge_cuda STATIC
  kernels.cu
  fast.cpp
)

target_include_directories(snapjudge_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
)

# Match TileLang numerics: fast math off for correctness-critical ops (attention
# softmax uses exp2 like the TileLang kernel).
set_target_properties(snapjudge_cuda PROPERTIES
  CUDA_ARCHITECTURES "80;86;89;90;100"
)
target_compile_options(snapjudge_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
)
