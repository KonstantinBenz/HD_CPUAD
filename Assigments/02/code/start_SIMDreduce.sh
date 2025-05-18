#!/usr/bin/env bash
#SBATCH -p rome
#SBATCH -w asc-rome02
#SBATCH --exclusive
#SBATCH --export=ALL      # <<< alle Login-Env-Vars werden vererbt
#SBATCH -o simd_reduce_output.txt

# Debug: prüfen, was wirklich ankommt
echo "LD_LIBRARY_PATH in job: $LD_LIBRARY_PATH" > debug_env.txt
ldd ./build/benchmarkSIMDreduce >> debug_env.txt 2>&1

# Jetzt läuft der Benchmark mit derselben Umgebung wie in
# deiner Login-Shell, weil du vorher "source load_env_CPUAD.sh" ausgeführt hast.
./build/benchmarkSIMDreduce >> simd_reduce_output.txt 2>&1