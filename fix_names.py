import os
import glob
import pathlib


def lines_that_start_with(string, fp):
    return [line for line in fp if line.startswith(string)]


def start_wandb():
    wandb_value = 888
    chosen_path = None
    folder_path = str(pathlib.Path().resolve()) + r"/scripts/"
    file_type = r'*out'
    files = glob.glob(folder_path + file_type)
    files.sort(key=os.path.getctime)
    last_files = files[:-10]
    for path in last_files:
        if int(path[path.rfind(r"/"):].find("wandb")) == 1:
            continue
        f = open(path, "r")
        #print("alines_that_start_with("wandb:", f))
        wandb_line = lines_that_start_with("wandb: Syncing run", f)
        print("path", path)
        print("wandb_line", wandb_line)
        f.close()
        wandb_value_list = [single_line[single_line.rfind("-") + 1:-1] for single_line in wandb_line]
        print("wandb_value_list",wandb_value_list)
        "_".join(wandb_value_list)
        if len(wandb_line) == 0:
            continue
        else:
            os.rename(path, folder_path + "wandb_" + "_".join(wandb_value_list) + ".out")

    print("chosen_path", chosen_path)
    # if chosen_path is not None:
    #    os.rename(chosen_path, folder_path + "wandb_" + str(wandb_value) + ".out")


if __name__ == '__main__':
    print("start!")
    start_wandb()
