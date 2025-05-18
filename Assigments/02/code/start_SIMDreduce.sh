#!/usr/bin/env bash
#SBATCH -p rome 
#SBATCH -w asc-rome02     # oder z. B. asc-rome03, falls Gruppe-ID % 4 = 3
#SBATCH --exclusive
#SBATCH -o simd_reduce_output.txt

#source load_env_CPUAD.sh
export LD_LIBRARY_PATH=/shares/asc-opt/spack/opt/spack/linux-debian12-x86_64_v2/gcc-14.2.0/intel-oneapi-compilers-2024.2.1-cs4fy4zo5nwzv5kqhki2qxjjw57pi32t/compiler/2024.2/lib:$LD_LIBRARY_PATH
export OMP_PLACES=numa_domains
export OMP_PROC_BIND=true

./build/benchmarkSIMDreduce