#!/usr/bin/python
from algorithms.ssl_algorithms import *
from algorithms.unsupervised_algorithms import *
from algorithms.mixmatch import MixMatch
from algorithms.remixmatch import ReMixMatch
from algorithms.uda import UDA, CTAUDA
from ss_clustering import SemiSupervisedClustering
from options import TrainingOptions
from easydict import EasyDict as e_dict
import time
from functools import partial
import os
import numpy as np
from collections import defaultdict
import torch
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import wandb

FLUSH_STEP = 20
TEACHER_PATH = 'teacher-temp'
NO_ESTIMATE_ITERS = 20


class SSClusteringRunner:
    def __init__(self, args):

        self.args = args
        self.args.state = None
        self.state_checkpoint_run_path = os.path.join(str(self.args.saving_dir_killable), "run",
                                                      "task" + str(self.args.slurm_task_id),
                                                      self.args.data_seeds + ".pkl")
        print("self.state_checkpoint_run_path", self.state_checkpoint_run_path)
        if os.path.exists(self.state_checkpoint_run_path):
            self.args.state = torch.load(self.state_checkpoint_run_path)
            self.load_state_args()

        if self.args.lab_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.lab_gpu
        self.ss_clust = self.get_ss_clust()

        if self.args.teacher_path is not None:  # change the labeled trainset to be the teacher set + the
            # K most confident pseudo labels of the teacher from each class. Not really used anymore.
            labeled_images, pseudo_labels = self.pseudo_label()
            self.ss_clust.labeled_trainset.data = labeled_images
            self.ss_clust.labeled_trainset.labels = pseudo_labels
        self.log_file = self.ss_clust.log_file
        self.s_epochs = 0
        self.us_epochs = 0
        self.other_us_stats = defaultdict(list)  # not used (noam)
        self.other_s_stats = defaultdict(list)  # not used (noam)
        self.classification_accs = []
        self.nmi_scores = []
        self.clustering_accs = []
        self.rotnet_losses = []
        self.rotnet_accs = []
        # TODO checkpoint
        if self.args.state is not None:
            self.load_train_members()

        self.graphs_dir = os.path.join(self.ss_clust.cur_dir, 'graphs')
        self.stats_dir = os.path.join(self.ss_clust.cur_dir, 'stats')
        if not os.path.exists(self.graphs_dir):
            os.mkdir(self.graphs_dir)
        if not os.path.exists(self.stats_dir):
            os.mkdir(self.stats_dir)

    def load_state_args(self):
        self.args = self.args.state["args"]

    def load_train_members(self):
        self.s_epochs = self.args.state["train_members"]["s_epochs"]
        self.us_epochs = self.args.state["train_members"]["us_epochs"]
        self.classification_accs = self.args.state["train_members"]["classification_accs"]
        self.nmi_scores = self.args.state["train_members"]["nmi_scores"]
        self.clustering_accs = self.args.state["train_members"]["clustering_accs"]
        self.rotnet_losses = self.args.state["train_members"]["rotnet_losses"]
        self.rotnet_accs = self.args.state["train_members"]["rotnet_accs"]

    def pseudo_label(self):
        teacher_ckpt = torch.load(self.args.teacher_path, map_location='cpu')
        teacher_args = e_dict(teacher_ckpt['args'])
        teacher_args.resume = self.args.teacher_path
        teacher_args.K = self.args.K
        teacher_args.estimate_perm = True
        teacher_args.rn = TEACHER_PATH
        teacher_args.data_dir = self.args.data_dir
        teacher_args.teacher_path = None
        teacher_args.debug = self.args.debug  # TODO remove
        teacher_args.save_pseudo = self.args.save_pseudo
        teacher = SemiSupervisedClustering(teacher_args)
        return teacher.labeled_trainset.data, teacher.labeled_trainset.labels

    def get_ss_clust(self):
        s_algo_to_class = {'fix_match': FixMatch, 'cta_clustering': CTAClustering, 's_fixmatch': SFixMatch,
                           'balanced_fixmatch': BalancedFixMatch, 'cta_fixmatch': CTAFixMatch,
                           'contrastive_fixmatch': ContrastiveFixMatch, 'mixmatch': MixMatch,
                           'uda': UDA, 'cta_uda': CTAUDA, 'remixmatch': ReMixMatch}
        us_algo_to_class = {'clustering': DeepClustering, 'us_fixmatch': USFixMatch, 'contrastive': USContrastive}

        class Dummy(s_algo_to_class[self.args.s_algo], us_algo_to_class[self.args.us_algo]):
            pass  # dummy class for inheritance

        return Dummy(args=e_dict(vars(self.args)))

    def train(self):
        print("self.args.iterations", self.args.iterations)
        if self.args.state is not None and not self.args.state["finished_rotnet"]:
            if self.args.rotnet_start_epochs > 0:  # rotnet warmup epochs.
                print("self.args.rotnet_start_epochs + 1", self.args.rotnet_start_epochs + 1)
                for i in range(1, self.args.rotnet_start_epochs + 1):
                    loss, acc = self.ss_clust.rotnet_epoch(optim=self.ss_clust.us_optim)
                    wandb.log({"rotnet_epoch loss": loss})
                    wandb.log({"rotnet_epoch acc": acc})
                    self.save_rotnet_stats(loss=loss, acc=acc, epoch=i, iteration=0,
                                           save_graphs=i == self.args.rotnet_start_epochs)
        print("finished rotnet_start_epochs")
        # self.save_state_checkpoint(False)
        if self.args.freeze > 0 and self.args.frozen_lr == 0:  # can be used for fine-tuning
            self.ss_clust.freeze_layers()
        initial_iteration = self.ss_clust.initial_iteration
        for i, iteration in enumerate(range(initial_iteration, initial_iteration + self.args.iterations)):

            if self.args.s_epochs[0] == 0 and i == 1 and self.args.estimate_perm:
                # used when we start by clustering, and afterwards want to estimate the permutation and train with
                # semi-supervised algorithms.
                self.ss_clust.initial_perm_estimation()
            self.s_epochs = self.args.s_epochs[min(i, len(self.args.s_epochs) - 1)]
            self.us_epochs = self.args.us_epochs[min(i, len(self.args.us_epochs) - 1)]
            self.log_file.write('start iteration number {}\n'.format(iteration + 1))
            last_iteration = iteration == initial_iteration + self.args.iterations - 1
            self.train_iteration(iteration=iteration, last_iteration=last_iteration)
            print("self.args.missing_labels", self.args.missing_labels, type(self.args.missing_labels))
            if self.args.missing_labels is not None and len(self.args.missing_labels) > 0:
                acc_score, validation_accuracy_missing_lables, validation_accuracy_appearing_lables,missing_labels_pred_count = self.ss_clust.eval_classifier(
                    missing_labels=self.args.missing_labels)
                wandb.log({"iteration validation accuracy missing lables": validation_accuracy_missing_lables})
                wandb.log({"iteration validation accuracy appearing lables": validation_accuracy_appearing_lables})
                print("missing_labels_pred_count",missing_labels_pred_count)
                wandb.log({"iteration missing labels pred count lables": missing_labels_pred_count})
            else:
                acc_score = self.ss_clust.eval_classifier()
            wandb.log({"iteration validation accuracy": acc_score})

            # self.save_state_checkpoint(True,iteration)
        self.log_file.close()
        print("final acc_score", acc_score)
        print("best_classifying_acc", self.ss_clust.best_classifying_acc)
        # self.finish_checkpoint()

    def save_state_checkpoint(self, finished_rotnet, iteration=0):

        train_members = {
            "s_epochs": self.s_epochs,
            "us_epochs": self.us_epochs,
            "classification_accs": self.classification_accs,
            "nmi_scores": self.nmi_scores,
            "clustering_accs": self.clustering_accs,
            "rotnet_losses": self.rotnet_losses,
            "rotnet_accs": self.rotnet_accs
        }
        ss_clustering_members = {
            "initial_iteration": iteration,
            "data_seeds": self.ss_clust.data_seeds,
            "labels_permutation": self.ss_clust.labels_permutation,
            "best_classifying_acc": self.ss_clust.best_classifying_acc,
            "best_clustering_acc": self.ss_clust.best_clustering_acc,
            "us_lowest_loss": self.ss_clust.us_lowest_loss,
            "s_lowest_loss": self.ss_clust.s_lowest_loss,
            "cur_dir": self.ss_clust.cur_dir
        }
        state = {
            "finished_rotnet": finished_rotnet,
            "args": self.args,
            "train_members": train_members,
            "ss_clustering_members": ss_clustering_members
        }
        torch.save(state, self.state_checkpoint_run_path)

    def finish_checkpoint(self):
        if os.path.exists(self.state_checkpoint_run_path):
            os.remove(self.state_checkpoint_run_path)

    def train_iteration(self, iteration, last_iteration=False):
        print("self.args.us_first",self.args.us_first)
        if self.s_epochs > 0 and not self.args.us_first:
            self.supervised_phase(iteration)
        if self.us_epochs > 0:
            pseudo_label = self.args.ul_to_l and not last_iteration
            self.unsupervised_phase(iteration=iteration, pseudo_label=pseudo_label)
            cur_iteration = iteration - self.ss_clust.initial_iteration + 1
            #  in the last 'NO_ESTIMATE_ITERS', don't estimate permutation.
            estimate_iteration = cur_iteration % self.args.estimate_freq == 0 and \
                                 self.args.iterations - cur_iteration > NO_ESTIMATE_ITERS
            # if 'estimate_perm' is on, we re-estimate the permutation every 'estimate_freq' iterations.
            if estimate_iteration and self.s_epochs > 0 and self.args.estimate_perm:
                self.ss_clust.estimate_labels_permutation()
            if self.args.us_first:
                if self.args.freeze > 0:
                    self.ss_clust.freeze_layers()
                self.supervised_phase(iteration)

    def supervised_phase(self, iteration):
        self.ss_clust.prepare_s_iteration()  # not really used.
        for epoch in range(1, self.s_epochs + 1):
            self.supervised_epoch(iteration=iteration, epoch=epoch)

        self.log_file.write('finished supervised phase\n')
        print('finished supervised phase')

    def unsupervised_phase(self, iteration, pseudo_label):
        self.ss_clust.prepare_us_iteration()  # not really used.
        for epoch in range(1, self.us_epochs + 1):
            self.unsupervised_epoch(iteration=iteration, epoch=epoch)

        print('finished unsupervised phase')
        self.log_file.write('finished unsupervised phase\n')
        if pseudo_label:
            self.ss_clust.pseudo_label()
            self.log_file.write('finished pseudo-labeling\n')
            print('finished pseudo-labeling')

    def supervised_epoch(self, iteration, epoch):
        torch.cuda.empty_cache()
        if epoch % FLUSH_STEP == 0:
            self.log_file.flush()
        save_graphs = epoch % FLUSH_STEP == 0 or epoch == self.s_epochs
        if self.args.s_rotnet_epoch:
            start = time.time()
            loss, acc = self.ss_clust.rotnet_epoch(optim=self.ss_clust.s_optim)
            print("rotnet epoch took {} seconds".format(time.time() - start))
            self.save_rotnet_stats(loss=loss, acc=acc, epoch=epoch, iteration=iteration + 1, save_graphs=save_graphs)

        start = time.time()
        rotnet_loss, rotnet_acc, other_stats = self.ss_clust.s_epoch(iteration, epoch, rotnet=self.args.s_rotnet_batch)
        print("semi-supervised epoch took {} seconds".format(time.time() - start))
        start = time.time()
        classification_acc = self.ss_clust.eval_classifier()
        other_stats['train_acc'] = self.ss_clust.eval_train()

        wandb.log({"labeled trainset classification accuracy": other_stats['train_acc']})
        wandb.log({"supervised epoch validation accuracy": classification_acc})
        print("evaluation took {} seconds".format(time.time() - start))
        self.save_supervised_stats(acc=classification_acc, other_stats=other_stats, epoch=epoch,
                                   iteration=iteration + 1, save_graphs=save_graphs)
        if self.args.s_rotnet_batch:
            self.save_rotnet_stats(loss=rotnet_loss, acc=rotnet_acc, epoch=epoch, iteration=iteration + 1,
                                   save_graphs=save_graphs)

    def unsupervised_epoch(self, iteration, epoch):
        torch.cuda.empty_cache()
        if epoch % FLUSH_STEP == 0:
            self.log_file.flush()
        save_graphs = epoch % FLUSH_STEP == 0 or epoch == self.us_epochs
        if self.args.us_rotnet_epoch:
            start = time.time()
            loss, acc = self.ss_clust.rotnet_epoch(optim=self.ss_clust.us_optim)
            print("rotnet epoch took {} seconds".format(time.time() - start))
            self.save_rotnet_stats(loss=loss, acc=acc, epoch=epoch, iteration=iteration + 1, save_graphs=save_graphs)

        if not self.args.only_rotnet:  # for ablation: using exactly the same pipeline, but obly with rotnet.
            start = time.time()
            rotnet_loss, rotnet_acc, other_stats = self.ss_clust.us_epoch(iteration, epoch,
                                                                          rotnet=self.args.us_rotnet_batch)
            print("unsupervised epoch took {} seconds".format(time.time() - start))
            if self.args.unsupervised_eval == 'unlabeled':
                print("eval_clustering, not eval_classifier")
                nmi_score, acc_score = self.ss_clust.eval_clustering()
            else:
                nmi_score = 0
                acc_score = self.ss_clust.eval_classifier(no_ema=True)
            wandb.log({"unsupervised epoch validation accuracy": acc_score})
            self.save_clustering_stats(nmi_score=nmi_score, acc_score=acc_score, epoch=epoch, iteration=iteration + 1,
                                       other_stats=other_stats, save_graphs=save_graphs)
            if self.args.us_rotnet_batch:
                self.save_rotnet_stats(loss=rotnet_loss, acc=rotnet_acc, epoch=epoch, iteration=iteration + 1,
                                       save_graphs=save_graphs)

    def save_clustering_stats(self, nmi_score, acc_score, epoch, iteration, other_stats, save_graphs=False):
        print("the nmi score in epoch number {} is: {}".format(epoch, nmi_score))
        print("the acc score in epoch number {} is: {}".format(epoch, acc_score))

        save_stat = partial(self.save_stat, iteration=iteration, epoch=epoch,
                            save_graph=save_graphs)

        save_stat(name='clustering_acc_score', value=acc_score, struct=self.clustering_accs)
        save_stat(name='nmi_score', value=nmi_score, struct=self.nmi_scores)
        for stat_name, stat_value in other_stats.items():
            save_stat(name=stat_name, value=stat_value, struct=self.other_s_stats[stat_name])

    def save_rotnet_stats(self, loss, acc, epoch, iteration, save_graphs=False):
        save_stat = partial(self.save_stat, iteration=iteration, epoch=epoch,
                            save_graph=save_graphs)
        save_stat(name='rotnet_train_loss', value=loss, struct=self.rotnet_losses)
        save_stat(name='rotnet_train_acc', value=acc, struct=self.rotnet_accs)

    def save_supervised_stats(self, acc, other_stats, epoch, iteration, save_graphs=False):
        print('the classification accuracy after {} iterations and {} epochs '
              'is: {}\n'.format(iteration, epoch, acc))

        save_stat = partial(self.save_stat, iteration=iteration, epoch=epoch, save_graph=save_graphs)
        save_stat(name='classification_acc', value=acc, struct=self.classification_accs)
        for stat_name, stat_value in other_stats.items():
            save_stat(name=stat_name, value=stat_value, struct=self.other_s_stats[stat_name])

    def save_stat(self, name, value, iteration, epoch, struct, save_graph=False):
        self.log_file.write('the {} after {} iterations and {} epochs is: {}\n'.format(name, iteration,
                                                                                       epoch, value))
        struct.append(value)
        np.save(file=os.path.join(self.stats_dir, '{}.npy'.format(name)), arr=struct)
        if save_graph:
            self.save_graph(values=struct, name=name)

    def save_graph(self, values, name):
        plt.plot(list(range(len(values))), values)
        plt.xlabel('epoch')
        plt.ylabel(name)
        plt.title(name)
        plt.savefig(os.path.join(self.graphs_dir, '{}.png'.format(name)))
        plt.close()
