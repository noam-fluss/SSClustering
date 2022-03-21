from ss_clustering import SemiSupervisedClustering, NUM_ROTATIONS
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import torch.nn.functional as F
from transforms import MODES
from utils.contrastive_loss import NTXentLoss
from sklearn.metrics import euclidean_distances
import lap
import wandb


class DeepClustering(SemiSupervisedClustering):
    """
    Our deep clustering algorithm which is explained thoroughly in the paper. This is the main unsupervised module we
    experimented with, and all the results in the paper are achieved with this module.
    """

    def __init__(self, args):
        super().__init__(args)

    def us_epoch(self, iteration, epoch, rotnet, **kwargs):
        mode = [MODES.TRAIN_MODE]
        if rotnet:
            mode.append(MODES.ROTNET_MODE)
        self.unlabeled_trainset.change_transform_mode(mode=mode)
        data_loader = DataLoader(self.unlabeled_trainset,
                                 batch_size=self.args.us_batch_size,
                                 shuffle=True,
                                 num_workers=self.args.workers,
                                 pin_memory=True)

        clustering_loss = 0.0
        rotnet_loss = 0.0
        rotnet_accuracy = 0.0
        num_switches = 0
        num_switches_from_missing = 0
        count_sum_targets_missing_label = 0
        count_sum_new_targets_missing_label = 0
        current_targets_missing_label = 0
        true_missing_labels_count = 0
        confident_missing_labels_value_count = 0
        true_other_labels_count = 0
        # Stream Training dataset with NAT
        for idx, images, label, nat in data_loader:

            if rotnet:
                (x, augmented), rotnet_batch = images
                rotnet_batch = torch.cat(rotnet_batch, dim=0).to(self.device)
                cur_rotnet_loss, cur_rotnet_acc = self.rotnet_batch(x=rotnet_batch, optim=self.us_optim)
                rotnet_loss += cur_rotnet_loss
                rotnet_accuracy += cur_rotnet_acc
            else:
                x, augmented = images

            targets = nat.numpy()
            x = x.to(self.device)
            if len(self.args.missing_labels) > 0:
                current_targets_missing_label = targets[:, self.args.missing_labels[0]].sum()
                count_sum_targets_missing_label += current_targets_missing_label
            wandb.log({"unsupervised current targets missing label": current_targets_missing_label})
            augmented = torch.cat(augmented, dim=0).to(self.device)
            cur_clustering_loss, batch_switches, batch_switches_from_missing, current_new_targets_missing_label, \
                    confident_missing_labels_values, true_other_labels_count_current = \
                            self.clustering_batch(x=x, augmented=augmented, targets=targets, indices=idx, label=label)
            count_sum_new_targets_missing_label += current_new_targets_missing_label
            true_other_labels_count += true_other_labels_count_current
            num_switches += batch_switches
            num_switches_from_missing += batch_switches_from_missing
            clustering_loss += cur_clustering_loss
            # TODO better coding
            for index in range(len(confident_missing_labels_values)):
                true_missing_labels_count += confident_missing_labels_values[index] in self.args.missing_labels
            confident_missing_labels_value_count += len(confident_missing_labels_values)

        wandb.log({"unsupervised batch targets missing label": count_sum_targets_missing_label})
        wandb.log({"unsupervised batch new targets missing label": count_sum_new_targets_missing_label})
        wandb.log({"unsupervised batch switch from missing labels": num_switches_from_missing})
        wandb.log({"unsupervised batch new targets missing label true labels count": true_missing_labels_count})
        wandb.log({"unsupervised batch new targets other label true labels count": true_other_labels_count})
        wandb.log({
            "unsupervised batch new targets missing label confident labels count": confident_missing_labels_value_count})
        if true_missing_labels_count == 0 and confident_missing_labels_value_count == 0:
            confident_missing_labels_value_count = 1

        wandb.log({"unsupervised batch new targets missing label true labels precision":
                       true_missing_labels_count / confident_missing_labels_value_count})

        if self.us_scheduler is not None:
            self.us_scheduler.step()
        if self.us_ema is not None:
            self.us_ema.update_buffer()

        clustering_loss /= len(self.unlabeled_trainset) * self.args.r
        rotnet_loss /= len(self.unlabeled_trainset) * NUM_ROTATIONS
        rotnet_accuracy /= len(self.unlabeled_trainset) * NUM_ROTATIONS
        us_stats = {'clustering_loss': clustering_loss, 'num_switches': num_switches}
        if clustering_loss < self.us_lowest_loss:
            self.us_lowest_loss = clustering_loss
            self.save_state(iteration=iteration, phase='us')
        self.save_state(iteration=iteration, phase='end_us')
        if len(self.args.missing_labels) > 0:
            unsupervised_batch_nat_missing_label = self.unlabeled_trainset.nat[:, self.args.missing_labels[0]].sum()
        else:
            unsupervised_batch_nat_missing_label = 0
        wandb.log({"unsupervised batch nat missing label": unsupervised_batch_nat_missing_label})
        return rotnet_loss, rotnet_accuracy, us_stats

    def clustering_batch(self, x, augmented, targets, indices, label):
        n_switches = 0
        true_other_labels_count = 0
        n_switches_from_missing = 0
        current_new_targets_missing_label = 0
        confident_missing_labels_values = []
        with torch.no_grad():
            if self.args.us_ema_teacher:
                output = F.normalize(self.us_ema.ema(x), dim=1, p=2)
            else:
                self.model.train(False)
                output = F.normalize(self.model(x), dim=1, p=2)
                self.model.train(True)
        output = output.detach().cpu().numpy()
        new_targets = np.copy(targets)
        real_targets = targets[np.sum(targets, axis=1) != 0]  # the non-zero targets.

        # finding the best assignment to targets according to the l2 distance of the model's output (projected to
        # the unit sphere) and all one-hot vectors.
        cost = euclidean_distances(output, real_targets)
        _, assignments, __ = lap.lapjv(cost, extend_cost=True)
        #
        assignments_cost = []
        cost_true_labels = np.array([])
        cost_true_missing_labels = np.array([])
        cost_true_appearing_labels = np.array([])

        # zero_choose = assignments[assignments == 0].index
        # confident_choose = cose[zero_choose][cose[zero_choose] > threshold].index
        for i in range(len(new_targets)):
            if assignments[i] == -1:  # means that the image hasn't got a target.
                new_targets[i] = np.zeros(self.num_classes)
            else:
                new_targets[i] = real_targets[assignments[i]]
                assignments_cost.append(cost[i][assignments[i]])
            no_real_target = np.logical_xor(np.any(new_targets[i]), np.any(targets[i]))  # whether either the old
            # target or new target are non-targets (zeros). Used to calculate the switches.
            target_switch = np.argmax(new_targets[i]) != np.argmax(targets[i])  # whether there was a cluster switch.
            n_switches += int(no_real_target or target_switch)
            if int(target_switch) and (np.argmax(new_targets[i]) not in self.args.missing_labels) and (np.argmax(
                    targets[i]) in self.args.missing_labels):
                n_switches_from_missing += 1

        for i in range(len(new_targets)):
            if np.argwhere(new_targets[i] == 1)[0][0] in self.args.missing_labels:
                if self.check_confident_missing(i, assignments, cost):
                    # label is the true label - checked
                    confident_missing_labels_values.append(label[i])
            if label[i] in new_targets.argmax(axis=1):
                cost_true_labels = np.append(cost_true_labels,
                                             cost[i][np.argwhere(new_targets.argmax(axis=1) == label[i].item())[0][0]])
                true_other_labels_count += len(cost_true_labels)
                if label[i].item in self.args.missing_labels:
                    cost_true_missing_labels = np.append(cost_true_missing_labels,
                                                         cost[i][
                                                             np.argwhere(new_targets.argmax(axis=1) == label[i].item())[
                                                                 0][0]])
                else:
                    cost_true_appearing_labels = np.append(cost_true_appearing_labels,
                                                           cost[i][
                                                               np.argwhere(
                                                                   new_targets.argmax(axis=1) == label[i].item())[0][
                                                                   0]])

        one_hot_pseudo_labels = np.eye(self.num_classes)[np.argmax(output, axis=1)]
        confidence = np.linalg.norm(output - one_hot_pseudo_labels, axis=1)
        new_augmented, y, missing_labels_weight = [], [], []

        for i in range(len(new_targets)):
            if np.sum(new_targets[i]) != 0:  # has target
                t = new_targets[i]
            elif confidence[i] < self.args.rho:  # high confidence sample receives a psuedo-target
                t = one_hot_pseudo_labels[i]
            else:  # the sample has no target and is not confident and hence is not processed.
                continue
            if np.argwhere(new_targets[i] == 1)[0][0] in self.args.missing_labels:
                current_weight = self.args.us_missing_labels_loss_weight
            else:
                current_weight = 1
            for j in range(self.args.r):  # include the r repetitions of the processed images.
                new_augmented.append(augmented[j * len(x) + i])
                y.append(t)
                missing_labels_weight.append(current_weight)

        new_augmented = torch.stack(new_augmented)
        output = F.normalize(self.model(new_augmented), dim=1, p=2)
        y = torch.tensor(y, dtype=torch.float, device=self.device)
        if int(self.args.us_missing_labels_loss_weight) == 1:
            loss = self.us_loss_fn(output, y)
        else:
            loss = self.us_loss_fn(output, y, missing_labels_weight)
        self.us_optim.zero_grad()
        loss.backward()
        self.us_optim.step()
        if self.us_ema is not None:
            self.us_ema.update_params()

        return loss.detach().item() * len(
            augmented), n_switches, n_switches_from_missing, current_new_targets_missing_label, confident_missing_labels_values, true_other_labels_count

    def wandb_cost(self, cost, cost_true_labels, cost_true_missing_labels, cost_true_appearing_labels, assignments_cost,
                   indices, new_targets):

        wandb.log(
            {"unsupervised reassignment min option cost - mean": np.sort(np.unique(cost, axis=1), axis=1)[:,
                                                                 [0]].flatten().mean()})
        wandb.log(
            {"unsupervised reassignment min option cost - std": np.sort(np.unique(cost, axis=1), axis=1)[:,
                                                                [0]].flatten().std()})
        wandb.log(
            {"unsupervised reassignment second min option cost - mean": np.sort(np.unique(cost, axis=1), axis=1)[:,
                                                                        [0]].flatten().mean()})
        wandb.log(
            {"unsupervised reassignment second option cost - std": np.sort(np.unique(cost, axis=1), axis=1)[:,
                                                                   [0]].flatten().std()})
        wandb.log({"unsupervised all true labels reassignment cost - mean": cost_true_labels.mean()})
        wandb.log({"unsupervised all true labels reassignment cost - std": cost_true_labels.std()})
        if len(cost_true_missing_labels) > 0:
            wandb.log({"unsupervised all true labels reassignment cost - mean": cost_true_missing_labels.mean()})
            wandb.log({"unsupervised all true labels reassignment cost - std": cost_true_missing_labels.std()})
        if len(cost_true_appearing_labels) > 0:
            wandb.log({"unsupervised missing true labels reassignment cost - mean": cost_true_appearing_labels.mean()})
            wandb.log({"unsupervised missing true labels reassignment cost - std": cost_true_appearing_labels.std()})
        wandb.log({"unsupervised appearing labels hungarian assignment cost - mean": np.array(assignments_cost).mean()})
        wandb.log({"unsupervised appearing labels hungarian assignment cost - std": np.array(assignments_cost).std()})
        self.unlabeled_trainset.update_targets(indices, new_targets)  # update the assignment to targets.
        if len(self.args.missing_labels) > 0:
            current_new_targets_missing_label = new_targets[:, self.args.missing_labels[0]].sum()
            wandb.log({"unsupervised new targets missing label": current_new_targets_missing_label})

    def check_confident_missing(self, i, assignments, cost):
        if cost[i][assignments[i]] <= np.sort(np.unique(cost, axis=1), axis=1)[:, [1]].flatten().mean():
            return True
        else:
            return False


class USFixMatch(SemiSupervisedClustering):
    """
    As explained in 'ssl_algorithms.py', this is an attempt to separate FixMatch to supervised and unsupervised phase.
    """

    def __init__(self, args):
        super().__init__(args)

    def us_epoch(self, iteration, epoch, rotnet, **kwargs):
        mode = [MODES.SS_MODE, MODES.ROTNET_MODE] if rotnet else [MODES.SS_MODE]
        self.unlabeled_trainset.change_transform_mode(mode=mode)
        train_loader = DataLoader(self.unlabeled_trainset,
                                  batch_size=self.args.us_batch_size,
                                  drop_last=False,
                                  num_workers=self.args.workers,
                                  pin_memory=False)
        # inverse function for going from normal labels to labels predicted by the model.
        labels_perm_inverse = np.where(self.labels_permutation == np.arange(self.num_classes)[:, None])[1]

        loss = torch.tensor(0.0, device=self.device)
        confident_samples_ratio = torch.tensor(0.0, device=self.device)
        pseudo_acc = torch.tensor(0.0, device=self.device)
        rotnet_loss = 0.0
        rotnet_acc = 0.0

        batch_mean_time = 0.0
        for i, data in enumerate(train_loader):
            if rotnet:
                idx, ((weak_batch, strong_batch), rotnet_batch), label, nat = data
                rotnet_batch = torch.cat(rotnet_batch, dim=0).to(self.device)
                cur_rotnet_loss, cur_rotnet_acc = self.rotnet_batch(x=rotnet_batch, optim=self.us_optim)
                rotnet_loss += cur_rotnet_loss
                rotnet_acc += cur_rotnet_acc
            else:
                idx, (weak_batch, strong_batch), label, nat = data

            if self.labels_permutation is not None:
                label = torch.from_numpy(labels_perm_inverse[label])
            start = time.time()
            cur_loss, num_confident, correct_pseudo = self.us_batch(weak_batch, strong_batch, label)
            batch_mean_time += (time.time() - start)
            loss += cur_loss
            confident_samples_ratio += num_confident
            pseudo_acc += correct_pseudo

        if self.us_scheduler is not None:
            self.us_scheduler.step()
        if self.us_ema is not None:
            self.us_ema.update_buffer()

        print("batch mean time took {} seconds".format(batch_mean_time / len(train_loader)))
        confident_samples_ratio = confident_samples_ratio.item()
        loss = loss.item() / max(confident_samples_ratio, 1)
        rotnet_samples = len(self.unlabeled_trainset) * NUM_ROTATIONS
        rotnet_loss /= rotnet_samples
        rotnet_acc /= rotnet_samples
        pseudo_acc = pseudo_acc.item() / max(confident_samples_ratio, 1)
        confident_samples_ratio /= len(self.unlabeled_trainset)
        us_stats = {'unlabeled_loss': loss, 'confidence_ratio': confident_samples_ratio,
                    'pseudo_acc': pseudo_acc}
        if loss < self.us_lowest_loss:
            self.us_lowest_loss = loss
            self.save_state(iteration=iteration, phase='us')
        self.save_state(iteration=iteration, phase='end_us')
        return rotnet_loss, rotnet_acc, us_stats

    def us_batch(self, weak_batch, strong_batch, label):
        with torch.no_grad():
            weak_batch = weak_batch.to(self.device)
            if self.args.us_ema_teacher:
                logits_weak = self.us_ema.ema(weak_batch)
            else:
                logits_weak = self.model(weak_batch)
            weak_probs = F.softmax(logits_weak.detach_(), dim=-1)
            max_probs, pseudo_labels = torch.max(weak_probs, dim=-1)
            idx = max_probs > self.args.confidence_threshold
            pseudo_labels = pseudo_labels[idx].detach()

        loss = torch.tensor(0.0)
        strong_batch = strong_batch[idx]
        if strong_batch.size(0) > 0:
            strong_batch = strong_batch.to(self.device)
            logits = self.model(strong_batch)
            loss = self.s_loss_fn(logits, pseudo_labels)

            self.us_optim.zero_grad()
            loss.backward()
            self.us_optim.step()

            if self.us_ema is not None:
                self.us_ema.update_params()

        confident_pseudo = strong_batch.size(0)
        correct_pseudo = torch.sum(pseudo_labels.detach() == label[idx].to(self.device))
        loss = loss.detach() * confident_pseudo
        return loss, confident_pseudo, correct_pseudo

    def prepare_us_iteration(self):
        for param in self.model.features[self.args.freeze:].parameters():
            param.requires_grad = True


class USContrastive(SemiSupervisedClustering):
    """
    Use the contrastive loss from the paper 'A Simple Framework for Contrastive Learning of Visual Representations'
    on the unlabeled data as the unsupervised phase.
    """

    def __init__(self, args):
        super().__init__(args)
        self.contrastive_loss = NTXentLoss(device=self.device, temperature=0.5, use_cosine_similarity=True)

    def us_epoch(self, iteration, epoch, rotnet, **kwargs):
        mode = [MODES.TRAIN_MODE]
        if rotnet:
            mode.append(MODES.ROTNET_MODE)
        self.unlabeled_trainset.change_transform_mode(mode=mode)
        data_loader = DataLoader(self.unlabeled_trainset,
                                 batch_size=self.args.us_batch_size,
                                 shuffle=True,
                                 num_workers=self.args.workers,
                                 pin_memory=False)

        contrastive_loss = 0.0
        rotnet_loss = 0.0
        rotnet_accuracy = 0.0

        for idx, images, label, nat in data_loader:
            if rotnet:
                (x1, x2), rotnet_batch = images
                rotnet_batch = torch.cat(rotnet_batch, dim=0).to(self.device)
                cur_rotnet_loss, cur_rotnet_acc = self.rotnet_batch(x=rotnet_batch, optim=self.us_optim)
                rotnet_loss += cur_rotnet_loss
                rotnet_accuracy += cur_rotnet_acc
            else:
                x1, x2 = images

            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            cur_contrastive_loss = self.us_batch(x1, x2)
            contrastive_loss += cur_contrastive_loss

        if self.us_scheduler is not None:
            self.us_scheduler.step()
        if self.us_ema is not None:
            self.us_ema.update_buffer()

        contrastive_loss /= (len(self.unlabeled_trainset) * 2)
        rotnet_loss /= len(self.unlabeled_trainset) * NUM_ROTATIONS
        rotnet_accuracy /= len(self.unlabeled_trainset) * NUM_ROTATIONS
        us_stats = {'contrastive_loss': contrastive_loss}
        if contrastive_loss < self.us_lowest_loss:
            self.us_lowest_loss = contrastive_loss
            self.save_state(iteration=iteration, phase='us')
        self.save_state(iteration=iteration, phase='end_us')
        return rotnet_loss, rotnet_accuracy, us_stats

    def us_batch(self, x1, x2):
        x1_logits = self.model(x1)
        x2_logits = self.model(x2)
        loss = self.contrastive_loss(x1_logits, x2_logits)

        self.us_optim.zero_grad()
        loss.backward()
        self.us_optim.step()

        if self.us_ema is not None:
            self.us_ema.update_params()

        return loss.detach() * x1.size(0) * 2
