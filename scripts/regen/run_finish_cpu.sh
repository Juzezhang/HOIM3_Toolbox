#!/bin/bash
#SBATCH -p sc-freecpu -A default
#SBATCH --cpus-per-task=14 --mem=48G --time=12:00:00
#SBATCH -o /simurgh2/users/juze/calibjoint/finish_cpu_%x_%j.log
SEQ="${SEQ:?}"
bash /simurgh2/users/juze/calibjoint/cutie_finish_seq.sh "$SEQ"
