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


def lines_that_start_with(string, fp):
    return [line for line in fp if line.startswith(string)]


def start_wandb():
    chosen_path = None
    wandb_value = int(wandb.run.name[wandb.run.name.rfind("-") + 1:])
    folder_path = str(pathlib.Path().resolve()) + r"/scripts/"
    file_type = r'*out'
    files = glob.glob(folder_path + file_type)
    files.sort(key=os.path.getctime)
    last_files = files[-2:]
    print("wandb_value", wandb_value)
    for path in last_files:
        f = open(path, "r")
        wandb_line = lines_that_start_with("wandb_run:", f)
        print("workers", lines_that_start_with("workers", f))
        print("path", path)
        print("wandb_line", wandb_line)
        f.close()
        print("finish!")
        if len(wandb_line) == 0:
            continue
        if wandb_value == int(wandb_line[0][11:-2]):
            chosen_path = path
            break
    print("chosen_path", chosen_path)
    if chosen_path is not None:
        os.rename(chosen_path, folder_path + "wandb_" + wandb.run.name[wandb.run.name.rfind("-") + 1:] + ".out")


if __name__ == '__main__':
    print("start!")

    args = TrainingOptions().parse()
    args = set_args_by_seed(args)
    config = vars(args)
    # TODO remove
    if args.semi_run_num_index == 2:
        print("this is args.semi_run_num_index")
        exit()
    wandb.init(project="new_run_missing_labels", entity="noam-fluss", config=config)
    print("wandb_run:", wandb.run.name[wandb.run.name.rfind("-") + 1:])

    #start_wandb()
    print("missing_labels", args.missing_labels)
    SSClusteringRunner(args).train()
