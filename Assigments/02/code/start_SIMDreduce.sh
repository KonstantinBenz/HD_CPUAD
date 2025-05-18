#!/usr/bin/env bash
#SBATCH -p rome
#SBATCH -w asc-rome02
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH -o simd_reduce_output.txt

# Hardcoded libstdc++ (für GLIBCXX_3.4.32) aus GCC 14.2
export LD_LIBRARY_PATH=/opt/asc/spack/opt/spack/linux-debian12-x86_64_v2/gcc-12.2.0/gcc-14.2.0-duw5n2gdhts3cjt6ikatgoh6g5qnibj3/lib64

# Hardcoded Intel libiomp5 Pfad (der, den du vorher per ldd hattest)
export LD_LIBRARY_PATH=/shares/asc-opt/spack/opt/spack/linux-debian12-x86_64_v2/gcc-14.2.0/intel-oneapi-compilers-2024.2.1-cs4fy4zo5nwzv5kqhki2qxjjw57pi32t/compiler/2024.2/lib:$LD_LIBRARY_PATH

# OpenMP-Settings
export OMP_PLACES=numa_domains
export OMP_PROC_BIND=true

# Debug-Check
ldd ./build/benchmarkSIMDreduce > debug_env.txt

# Benchmark ausführen
./build/benchmarkSIMDreduce >> simd_reduce_output.txt 2>&1