#!/usr/bin/env bash
#SBATCH -p rome 
#SBATCH -w asc-rome02     # oder z. B. asc-rome03, falls Gruppe-ID % 4 = 3
#SBATCH --exclusive
#SBATCH --export=ALL 
#SBATCH -o simd_reduce_output.txt

#source load_env_CPUAD.sh
export OMP_PLACES=numa_domains
export OMP_PROC_BIND=true

./build/benchmarkSIMDreduce