#!/bin/bash
#SBATCH -c 6
#SBATCH --mem=8g
#SBATCH --time=2-0
## 10g selects a better GPU, if we're paying for it.
#SBATCH --gres=gpu:1,vmem:10g
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8
#SBATCH --array=0-5
#SBATCH --exclude=ampere-01,gsm-04,creek-04
#SBATCH --killable


function ind2sub() {
    local idx="$1"   # Save first argument in a variable
    shift            # Shift all arguments to the left (original $1 gets lost)
    local shape=("$@") # Rebuild the array with rest of arguments
    local cur_idx=$(($idx))  #zero base

    num_dims=${#shape[@]}  # returns the length of an array
    for ((i=0; i<$num_dims; i++))
    do
        cur_dim=${shape[$i]}
        idxes[$i]=$(($cur_idx%$cur_dim))
        # echo ${idxes[$i]}
        # echo $(($cur_idx%$i))
        local cur_idx=$(($cur_idx/$cur_dim))
    done
    # echo $idx
}

function cumprod() {
    local arr=("$@") # Rebuild the array with rest of arguments
    local prod=1
    for ((i=0; i<${#arr[@]}; i++))
    do
        ((prod *= ${arr[$i]}))
    done
    echo $prod
}

function shape_from_arrays() {
    local arr=("$@") # Rebuild the array with rest of arguments
    for ((i=0; i<${#arr[@]}; i++))
    do
        local -n cur_array=${arr[$i]}   # -n for declaring an array
        shape+=(${#cur_array[@]})
    done
}
module load torch


dir=/cs/labs/daphna/noam.fluss/project/SSClustering/
cd $dir
source /cs/labs/daphna/avihu.dekel/env/bin/activate



SEEDS=(1 2 3)
NUM_SAMPLES=(10 30 50)

shape_from_arrays SEEDS NUM_SAMPLES
ind2sub ${SLURM_ARRAY_TASK_ID} "${shape[@]}"

seed=${SEEDS[${idxes[0]}]}
n_labels=${NUM_SAMPLES[${idxes[1]}]}

python3 train.py --dataset cifar10 --data_seeds ${seed} --n_labels ${n_labels} --crop_size 32 --us_rotnet_epoch --milestones 100 --s_ema_eval --rn 10nov_cifar10_${n_labels}_imbalanced --iterations 80 --imbalanced
python3 train.py --dataset cifar10 --data_seeds $seed --n_labels $n_labels --crop_size 32 --us_rotnet_epoch --milestones 100 --s_ema_eval --rn cifar10-labels=$n_labels,partition=$seed
