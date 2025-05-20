#!/usr/bin/env bash
#SBATCH -p rome
#SBATCH -w asc-rome00
#SBATCH --exclusive
#SBATCH -o ex03_benchmark.out

export OMP_PLACES=numa_domains
export OMP_PROC_BIND=true

cd /media/oldhome/kbenz/git/HD_CPUAD/Assigments/03/code/build

#echo "Starting REDUCE benchmark..."
#./reduceVbenchmarkUnroll > reduce_results.csv

echo "Starting TRANSFORM benchmark..."
./transformVbenchmarkUnroll > transform_results.csv

echo "All benchmarks completed."