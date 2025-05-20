#!/usr/bin/env bash
#SBATCH -p rome
#SBATCH -w asc-rome02
#SBATCH --exclusive
#SBATCH -o ex03_benchmark.out

export OMP_PLACES=numa_domains
export OMP_PROC_BIND=true

cd /media/oldhome/kbenz/git/HD_CPUAD/Assigments/03/code/build

#echo "Starting REDUCE benchmark..."
#ls -lh ./reduceVbenchmarkUnroll
#./reduceVbenchmarkUnroll > reduce_results.csv 2>&1

echo "Starting TRANSFORM benchmark..."
ls -lh ./transformVbenchmarkUnroll
./transformVbenchmarkUnroll > transform_results.csv 2>&1

echo "All benchmarks completed."