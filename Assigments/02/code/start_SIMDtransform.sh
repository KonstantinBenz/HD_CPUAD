#!/usr/bin/env bash
#SBATCH -p rome 
#SBATCH -w asc-rome02
#SBATCH --exclusive
#SBATCH -o simd_transform_output.txt

source load_env_CPUAD.sh

export OMP_PLACES=numa_domains
export OMP_PROC_BIND=true

./build/benchmarkSIMDtransform
