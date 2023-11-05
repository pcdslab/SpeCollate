import random as rand
import sys
import timeit
# from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import progressbar
import wandb
# from tqdm import tqdm

from ..atlesconfig import config

rand.seed(37)

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []
mse_weight = config.get_config(section="ml", key="mse_weight")
ce_weight_clv = config.get_config(section="ml", key="ce_weight_clv")
ce_weight_mod = config.get_config(section="ml", key="ce_weight_mod")
divider = ce_weight_clv + mse_weight


def train(model, device, train_loader, mse_loss, ce_loss, optimizer, scaler, epoch):
    model.train()

    accurate_cleavs = accurate_mods = 0
    accurate_labels_0 = accurate_labels_1 = accurate_labels_2 = 0
    all_labels = 0
    tot_mse_loss = tot_ce_clv_loss = tot_ce_mod_loss = tot_loss = 0

    # pbar = tqdm(train_loader, file=sys.stdout)
    # pbar.set_description('Training...')
    for data in train_loader:
        data[0] = data[0].to(device)  # spec
        # data[1] = data[1].to(device)  # intensities
        data[1] = data[1].to(device)  # charge gray-mass
        data[2] = data[2].to(device)  # pep lens
        data[3] = data[3].to(device)  # modifications
        data[4] = data[4].to(device)  # missed cleavages

        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            input_mask = data[0] == 0
            # input_mask = 0
            lens, cleavs, mods = model(data[0], data[1], input_mask)
            lens = lens.squeeze()
            # cleavs = cleavs.squeeze()
            # print(len(cleavs))
            # print(torch.min(data[5]), torch.max(data[5]))
            mse_loss_val = mse_loss(lens, data[2])
            ce_clv_loss_val = ce_loss(cleavs, data[4])
            ce_mod_loss_val = ce_loss(mods, data[3])
            loss = (mse_weight * mse_loss_val + ce_weight_clv * ce_clv_loss_val +
                    ce_weight_mod * ce_mod_loss_val)  # / divider
            # loss = sum(loss_lst) / len(loss_lst)
            # loss = mse_loss_val
            tot_mse_loss += float(mse_loss_val)
            tot_ce_clv_loss += float(ce_clv_loss_val)
            tot_ce_mod_loss += float(ce_mod_loss_val)
            tot_loss += float(loss)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        # accurate_labels += multi_acc(lens, data[2])
        accurate_labels_0 += mse_acc(lens, data[2], err=0)
        accurate_labels_1 += mse_acc(lens, data[2], err=1)
        accurate_labels_2 += mse_acc(lens, data[2], err=2)
        accurate_cleavs += multi_acc(cleavs, data[4])
        accurate_mods += multi_acc(mods, data[3])
        all_labels += len(lens)
        # p_bar.update(idx)

    # accuracy = 100. * float(accurate_labels) / all_labels
    accuracy_0 = 100. * float(accurate_labels_0) / all_labels
    accuracy_1 = 100. * float(accurate_labels_1) / all_labels
    accuracy_2 = 100. * float(accurate_labels_2) / all_labels
    accuracy_cleavs = 100. * float(accurate_cleavs) / all_labels
    accuracy_mods = 100. * float(accurate_mods) / all_labels

    wandb.log({"Train loss": tot_loss / len(train_loader)}, step=epoch)
    wandb.log({"Train Accuracy Margin-0": accuracy_0}, step=epoch)
    wandb.log({"Train Accuracy Margin-1": accuracy_1}, step=epoch)
    wandb.log({"Train Accuracy Margin-2": accuracy_2}, step=epoch)
    wandb.log({"Train Accuracy Missed Cleavages": accuracy_cleavs}, step=epoch)
    wandb.log({"Train Accuracy Modifications": accuracy_mods}, step=epoch)

    print('Train accuracy Margin 0:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_0,
          all_labels, accuracy_0, tot_mse_loss / len(train_loader)))
    print('Train accuracy Margin 1:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_1,
          all_labels, accuracy_1, tot_mse_loss / len(train_loader)))
    print('Train accuracy Margin 2:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_2,
          all_labels, accuracy_2, tot_mse_loss / len(train_loader)))
    print('Train accuracy Clvs:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_cleavs,
          all_labels, accuracy_cleavs, tot_ce_clv_loss / len(train_loader)))
    print('Train accuracy Mods:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_mods,
          all_labels, accuracy_mods, tot_ce_mod_loss / len(train_loader)))
    print('Total Loss:\t{}'.format(tot_loss / len(train_loader)))
    return loss


def test(model, device, test_loader, mse_loss, ce_loss, epoch):
    model.eval()

    with torch.no_grad():
        accurate_cleavs = accurate_mods = 0
        accurate_labels_0 = accurate_labels_1 = accurate_labels_2 = 0
        all_labels = 0
        tot_mse_loss = tot_ce_clv_loss = tot_ce_mod_loss = tot_loss = 0
        pred_lens, pred_cleavs, pred_mods = [], [], []
        labl_lens, labl_cleavs, labl_mods = [], [], []
        # with progressbar.ProgressBar(max_value=len(train_loader)) as p_bar:
        for data in test_loader:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                data[0] = data[0].to(device)  # spec
                # data[1] = data[1].to(device) # intensities
                data[1] = data[1].to(device)  # charge gray-mass
                data[2] = data[2].to(device)  # pep lens
                data[3] = data[3].to(device)  # modifications
                data[4] = data[4].to(device)  # missed cleavages

                input_mask = data[0] == 0
                # input_mask = 0
                lens, cleavs, mods = model(data[0], data[1], input_mask)
                lens = lens.squeeze()
                
                pred_lens.extend(lens.tolist())
                pred_cleavs.extend(cleavs.tolist())
                pred_mods.extend(mods.tolist())
                
                labl_lens.extend(data[2].tolist())
                labl_cleavs.extend(data[4].tolist())
                labl_mods.extend(data[3].tolist())
                # cleavs = cleavs.squeeze()
                mse_loss_val = mse_loss(lens, data[2])
                ce_clv_loss_val = ce_loss(cleavs, data[4])
                ce_mod_loss_val = ce_loss(mods, data[3])
                loss = (mse_weight * mse_loss_val + ce_weight_clv * ce_clv_loss_val +
                        ce_weight_mod * ce_mod_loss_val)  # / divider
                # loss = sum(loss_lst) / len(loss_lst)
                # loss = mse_loss_val
                tot_mse_loss += float(mse_loss_val)
                tot_ce_clv_loss += float(ce_clv_loss_val)
                tot_ce_mod_loss += float(ce_mod_loss_val)
                tot_loss += float(loss)

            # accurate_labels += multi_acc(lens, data[2])
            accurate_labels_0 += mse_acc(lens, data[2], err=0)
            accurate_labels_1 += mse_acc(lens, data[2], err=1)
            accurate_labels_2 += mse_acc(lens, data[2], err=2)
            accurate_cleavs += multi_acc(cleavs, data[4])
            accurate_mods += multi_acc(mods, data[3])
            all_labels += len(lens)

        # accuracy = 100. * float(accurate_labels) / all_labels
        accuracy_0 = 100. * float(accurate_labels_0) / all_labels
        accuracy_1 = 100. * float(accurate_labels_1) / all_labels
        accuracy_2 = 100. * float(accurate_labels_2) / all_labels
        accuracy_cleavs = 100. * float(accurate_cleavs) / all_labels
        accuracy_mods = 100. * float(accurate_mods) / all_labels

        wandb.log({"Test loss": tot_loss / len(test_loader)}, step=epoch)
        wandb.log({"Test Accuracy Margin-0": accuracy_0}, step=epoch)
        wandb.log({"Test Accuracy Margin-1": accuracy_1}, step=epoch)
        wandb.log({"Test Accuracy Margin-2": accuracy_2}, step=epoch)
        wandb.log({"Test Accuracy Missed Cleavages": accuracy_cleavs}, step=epoch)
        wandb.log({"Test Accuracy Modifications": accuracy_mods}, step=epoch)

        print('Test accuracy Margin 0:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_0,
              all_labels, accuracy_0, tot_mse_loss / len(test_loader)))
        print('Test accuracy Margin 1:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_1,
              all_labels, accuracy_1, tot_mse_loss / len(test_loader)))
        print('Test accuracy Margin 2:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_2,
              all_labels, accuracy_2, tot_mse_loss / len(test_loader)))
        print('Test accuracy Clvs:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_cleavs,
              all_labels, accuracy_cleavs, tot_ce_clv_loss / len(test_loader)))
        print('Test accuracy Mods:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_mods,
              all_labels, accuracy_mods, tot_ce_mod_loss / len(test_loader)))
        print('Total Loss:\t{}'.format(tot_loss / len(test_loader)))
        return loss, torch.tensor(pred_lens), torch.tensor(labl_lens), torch.tensor(pred_cleavs),\
            torch.tensor(labl_cleavs), torch.tensor(pred_mods), torch.tensor(labl_mods)


# TODO: change it. taken from
# https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
# accessed: 09/18/2020
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float().sum()

    return correct_pred


def mse_acc(y_pred, y_test, err=0):
    correct_pred = (torch.abs(torch.round(y_pred) - torch.round(y_test)) <= err).float().sum()

    return correct_pred
