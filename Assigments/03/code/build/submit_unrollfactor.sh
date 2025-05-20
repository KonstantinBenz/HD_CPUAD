#!/usr/bin/env bash
#SBATCH -p rome 
#SBATCH -w asc-rome00
#SBATCH --exclusive
#SBATCH -o res/program_256.txt

export OMP_PLACES=numa_domains # threads for single-pass scan, otherwise numa_domains
export OMP_PROC_BIND=true

./changingUnrollFactor/program_256
