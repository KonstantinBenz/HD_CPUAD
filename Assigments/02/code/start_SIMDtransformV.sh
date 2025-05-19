#!/usr/bin/env bash
#SBATCH -p rome
#SBATCH -w asc-rome02
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH -o simd_transformV_output.txt

export OMP_PLACES=numa_domains
export OMP_PROC_BIND=true

./build/benchmarkSIMDtransformV