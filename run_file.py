from train import TrainingOptions, SSClusteringRunner
import wandb
import os
import glob
import os.path
import pathlib


def set_args_by_seed(args):
    args.run_num_index = int(args.data_seeds) // 9
    args.semi_run_num_index = int(args.data_seeds) % 9
    args.data_seeds = str(args.run_num_index)
    if args.semi_run_num_index > 0:
        args.missing_labels = [(args.semi_run_num_index - 1) % 3]
        args.us_epochs = [((args.semi_run_num_index - 1) // 3) + 1]
        args.us_first = True
    return args


def start_wandb():
    wandb.init(project="new_run_missing_labels", entity="noam-fluss", config=config)
    folder_path = str(pathlib.Path().resolve()) + r"/scripts/"
    file_type = r'*out'
    files = glob.glob(folder_path + file_type)
    max_file = max(files, key=os.path.getctime)
    os.rename(max_file, folder_path + "wandb_" + wandb.run.name[wandb.run.name.rfind("-") + 1:] + ".out")


if __name__ == '__main__':
    print("start!")
    args = TrainingOptions().parse()
    config = vars(args)
    args = set_args_by_seed(args)
    print("args", args)
    start_wandb()
    print("missing_labels", args.missing_labels)
    SSClusteringRunner(args).train()
