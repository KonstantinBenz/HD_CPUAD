#!/usr/bin/env bash
#SBATCH -p rome
#SBATCH -w asc-rome02
#SBATCH --exclusive
#SBATCH -o simd_reduce_output.txt

# 1. Spack aktivieren (wichtig, sonst kennt die Shell 'spack' nicht!)
source /opt/asc/spack/share/spack/setup-env.sh

# 2. Umgebung wie interaktiv vollständig laden
source load_env_CPUAD.sh

# 3. Debug-Output zur Kontrolle
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
ldd ./build/benchmarkSIMDreduce

# 4. Benchmark starten
./build/benchmarkSIMDreduce
