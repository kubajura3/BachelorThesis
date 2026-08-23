import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Portable default: every major NVIDIA architecture from Pascal (2016) onward,
# plus PTX at compute_90 so a future GPU can JIT from it. Building all of these
# takes several minutes.
#
# For a fast local build, set TORCH_CUDA_ARCH_LIST (e.g. "7.5" for a GTX 16xx /
# RTX 20xx, or "native"). That has to disable the list below rather than sit
# alongside it: torch.utils.cpp_extension._get_cuda_arch_flags() bails out and
# adds nothing of its own as soon as any user-supplied nvcc flag contains
# "arch", so leaving these in would silently ignore the environment variable and
# rebuild every architecture anyway.
if os.getenv("TORCH_CUDA_ARCH_LIST"):
    NVCC_GENCODE = []                            # torch expands the env var instead
else:
    NVCC_GENCODE = [
        "-gencode=arch=compute_60,code=sm_60",       # Pascal (GTX 10xx, Tesla P100)
        "-gencode=arch=compute_61,code=sm_61",       # Pascal (GTX 10xx desktop)
        "-gencode=arch=compute_70,code=sm_70",       # Volta (V100)
        "-gencode=arch=compute_75,code=sm_75",       # Turing (RTX 20xx, GTX 16xx, T4)
        "-gencode=arch=compute_80,code=sm_80",       # Ampere (A100)
        "-gencode=arch=compute_86,code=sm_86",       # Ampere (RTX 30xx)
        "-gencode=arch=compute_89,code=sm_89",       # Ada Lovelace (RTX 40xx)
        "-gencode=arch=compute_90,code=sm_90",       # Hopper (H100)
        "-gencode=arch=compute_90,code=compute_90",  # PTX for future GPUs (JIT compiled)
    ]

setup(
    name="srbd_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="srbd_cuda_ext",
            sources=[
                "src/srbd_ext.cpp",
                "src/srbd_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",           # safe: SRBD is corrected by alpha-alignment each step
                    "--expt-relaxed-constexpr",
                ] + NVCC_GENCODE,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
