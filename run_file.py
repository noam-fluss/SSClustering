from train import TrainingOptions, SSClusteringRunner
import wandb
import os
import glob
import os.path
import pathlib


def set_args_by_seed(args):
    args.data_seeds = str(int(args.data_seeds))
    args.run_num_index = int(args.data_seeds) // 9
    args.semi_run_num_index = int(args.data_seeds) % 9
    args.data_seeds = str(args.run_num_index)
    if args.semi_run_num_index > 0:
        args.n_labels = 36
        args.missing_labels = [(args.semi_run_num_index - 1) % 3]
        args.us_epochs = [((args.semi_run_num_index - 1) // 3) + 1]
        args.us_first = True
    return args


if __name__ == '__main__':
    print("start!")

    args = TrainingOptions().parse()
    args = set_args_by_seed(args)
    config = vars(args)
    wandb.init(project="new_run_missing_labels", entity="noam-fluss", config=config)
    print("wandb_run:", wandb.run.name[wandb.run.name.rfind("-") + 1:])

    print("missing_labels", args.missing_labels)
    SSClusteringRunner(args).train()
