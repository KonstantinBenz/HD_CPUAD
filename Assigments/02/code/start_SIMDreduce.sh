#!/usr/bin/env bash
#SBATCH -p rome
#SBATCH -w asc-rome02   # oder 00/01/03 je nach groupID % 4
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH -o simd_reduce_output.txt

export OMP_PLACES=numa_domains
export OMP_PROC_BIND=true

./build/benchmarkSIMDreduce