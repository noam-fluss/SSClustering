#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 4
#SBATCH --time=2-0
#SBATCH --gres=gpu:1,vmem:10g
#SBATCH --mail-user=noam.fluss@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
## 10g selects a better GPU, if we're paying for it.
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8
#SBATCH --array=0-1
#SBATCH --exclude=ampere-01,gsm-04

module load torch

dir=/cs/labs/daphna/noam.fluss/project/SSClustering/

cd $dir

source /cs/labs/daphna/avihu.dekel/env/bin/activate

python3 train.py --dataset cifar10 --n_labels 40 --crop_size 32 --us_rotnet_epoch --milestones 100 --s_ema_eval \
        --rn test1 --data_seeds ${SLURM_ARRAY_TASK_ID} \
        --missing_labels 0 --iterations 5 --s_epochs 2 --rotnet_start_epochs 2
