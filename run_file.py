from train import TrainingOptions,SSClusteringRunner
import wandb

def set_args_by_seed(args):
    if int(args.data_seeds) >= 0:
        args.missing_labels = [0]
        #args.us_epochs = [2]
        args.us_first = True
    if int(args.data_seeds) >= 3:
        args.us_epochs = [int(args.data_seeds) - 1]
    return args


if __name__ == '__main__':
    print("start")
    args = TrainingOptions().parse()
    config = vars(args)
    args = set_args_by_seed(args)
    print("args", args)
    wandb.init(project="new_run_missing_labels", entity="noam-fluss", config=config)

    SSClusteringRunner(args).train()
