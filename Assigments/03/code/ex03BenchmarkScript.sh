#!/bin/bash
#SBATCH --job-name=ex03_benchmark
#SBATCH --output=ex03_benchmark_%j.out
#SBATCH --error=ex03_benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=exercise
#SBATCH --time=00:30:00

# Load environment (falls nicht automatisch)
source ../load_env_CPUAD.sh

# Optional: in build-Verzeichnis wechseln
cd "$(dirname "$0")"

echo "Running reduce benchmarks..."
./reduceVbenchmarkUnroll > reduce_results.csv

echo "Running transform benchmarks..."
./transformVbenchmarkUnroll > transform_results.csv

echo "Benchmarking done."