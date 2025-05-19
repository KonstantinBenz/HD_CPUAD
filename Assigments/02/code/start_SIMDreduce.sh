#!/usr/bin/env bash
#SBATCH -p rome
#SBATCH -w asc-rome02
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH -o simd_reduce_output.txt


# OpenMP-Settings
export OMP_PLACES=numa_domains
export OMP_PROC_BIND=true

# Debug-Check
ldd ./build/benchmarkSIMDreduce > debug_env.txt

# Benchmark ausführen
./build/benchmarkSIMDreduce >> simd_reduce_output.txt 2>&1