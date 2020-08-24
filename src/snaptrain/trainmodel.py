import random as rand
import atexit
#from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import progressbar

from src.snapconfig import config
from src.snaptrain import process

rand.seed(37)

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []


# def exit_handler():
#     plt.figure()
#     plt.plot(train_accuracy)
#     plt.plot(test_accuracy)
#     plt.xlabel("epoch")
#     plt.ylabel("accuracy")
#     plt.savefig("accuracy.png", dpi=600)

#     plt.figure()
#     plt.plot(train_loss)
#     plt.plot(test_loss)
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.savefig("loss.png", dpi=600)


# atexit.register(exit_handler)


def train(model, device, train_loader, triplet_loss, optimizer):
    model.train()
    batch_size = config.get_config(section='ml', key='batch_size')
    h = model.module.init_hidden(batch_size)

    accurate_labels = 0
    all_labels = 0
    l = 0
    with progressbar.ProgressBar(max_value=len(train_loader)) as bar:
        for idx, data in enumerate(train_loader):
            h = tuple([e.data for e in h])
            q_len = len(data[0])
            p_len = len(data[1])
            d_len = len(data[2])
            if p_len > d_len:
                seq_len = config.get_config(section='ml', key='pep_seq_len')
                zero_pad = torch.zeros(p_len - d_len, seq_len, dtype=torch.long)
                data[2] = torch.cat((data[2], zero_pad))
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)
            counts = data[3]
            
            optimizer.zero_grad()
            
            Q, P, D, h = model(data[:-1], h)

            loss, QxPD = snap_loss2(counts, P, Q, D[:d_len], triplet_loss, device)

            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            optimizer.step()
            
            seq, _ = get_index(counts, q_len)
            seq = torch.LongTensor(seq).to(device)
            # QxP contains the distance of each spectrum from each peptide.
            accurate_labels = accurate_labels + torch.sum(QxPD.argmin(1) == seq)
            
            all_labels = all_labels + len(Q)
            bar.update(idx)
    
    accuracy = 100. * float(accurate_labels) / all_labels
    train_accuracy.append(accuracy)
    train_loss.append(loss)
    print('Train accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))
    

def test(model, device, test_loader, triplet_loss):
    model.eval()
    
    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        loss = 0

        batch_size = config.get_config(section='ml', key='batch_size')
        h = model.module.init_hidden(batch_size)
        
        for data in test_loader:
            h = tuple([e.data for e in h])
            q_len = len(data[0])
            p_len = len(data[1])
            d_len = len(data[2])
            if p_len > d_len:
                seq_len = config.get_config(section='ml', key='pep_seq_len')
                zero_pad = torch.zeros(p_len - d_len, seq_len, dtype=torch.long)
                data[2] = torch.cat((data[2], zero_pad))
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)
            counts = data[3]
            
            Q, P, D, h = model(data[:-1], h)

            loss, QxPD = snap_loss2(counts, P, Q, D[:d_len], triplet_loss, device)
            
            seq, _ = get_index(counts, q_len)
            seq = torch.LongTensor(seq).to(device)
            # QxP contains the distance of each spectrum from each peptide.
            accurate_labels = accurate_labels + torch.sum(QxPD.argmin(1) == seq)
            
            all_labels = all_labels + len(Q)
                
        accuracy = 100. * float(accurate_labels) / all_labels
        test_accuracy.append(accuracy)
        test_loss.append(loss)
        print('Test accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))


def get_masks(counts, p_len, q_len):
    rows, cols = get_index(counts, q_len)

    PQ_mask = torch.zeros(p_len, q_len)
    PQ_mask[rows, cols] = 1.
    
    QQ_mask = torch.zeros(q_len, q_len)
    QQ_mask[cols, :] = PQ_mask[rows, :]
    
    return PQ_mask, QQ_mask


def get_index(counts, q_len):
    rows = []
    for i, x in enumerate(counts):
        rows.extend([i]*x)
    cols = range(q_len)
    return rows, cols


def snap_loss(counts, Q, P, triplet_loss, device):
    """Mine the hardest sextuplets.""" 
    QxQ = process.pairwise_distances(Q)     # calculate distance matrix for spectra
    PxP = process.pairwise_distances(P)     # calculate distance matrix for peptides
    QxP = process.pairwise_distances(Q, P)  # calculate distance matrix for spectra-peptides
    
    # Get maskes
    p_len, q_len = len(P), len(Q)
    PQ_mask, QQ_mask = get_masks(counts, p_len, q_len)
    PQ_mask, QQ_mask = PQ_mask.to(device), QQ_mask.to(device)

    # Mine hardest positives
    PxQ_max = Q[(torch.transpose(QxP, 0, 1) * PQ_mask).max(1).indices]
    QxP_max = P[(QxP * torch.transpose(PQ_mask, 0, 1)).max(1).indices]

    # Mine hardest negatives
    QQ_neg_mask = 1 - QQ_mask
    QQ_neg_mask_clone = QQ_neg_mask.clone()
    QxQ_masked = QxQ * QQ_neg_mask
    QxQ_masked[QQ_neg_mask_clone < 0.1] = float("inf")
    QxQ_min = Q[QxQ_masked.min(1).indices]              # nearest spectrum for each spectrum

    PxP.fill_diagonal_(float("inf"))
    PxP_min = P[PxP.min(1).indices]                     # nearest peptide for each peptide

    PQ_neg_mask = (1 - PQ_mask)
    PQ_neg_mask_clone = PQ_neg_mask.clone()
    PxQ_masked = torch.transpose(QxP, 0, 1) * PQ_neg_mask
    PxQ_masked[PQ_neg_mask_clone < 0.1] = float("inf")
    PxQ_min = Q[PxQ_masked.min(1).indices]              # nearest spectrum for each peptide

    QP_neg_mask = torch.transpose(PQ_neg_mask, 0, 1)
    QP_neg_mask_clone = QP_neg_mask.clone()
    QxP_masked = QxP * QP_neg_mask
    QxP_masked[QP_neg_mask_clone < 0.1] = float("inf")
    QxP_min = P[QxP_masked.min(1).indices]              # nearest peptide for each spectrum

    # Calculate the loss
    loss  = triplet_loss(Q, QxP_max, QxQ_min)   # spectrum-spectrum negatives
    loss += triplet_loss(Q, QxP_max, QxP_min)   # spectrum-peptide negatives
    loss += triplet_loss(P, PxQ_max, PxP_min)   # peptide-peptide negatives
    loss += triplet_loss(P, PxQ_max, PxQ_min)   # peptide-spectrum negatives
    
    loss = loss / 4

    return loss, QxP


def snap_loss2(counts, P, Q, D, triplet_loss, device):
    """Mine the hardest sextuplets."""
    PxP = process.pairwise_distances(P)     # calculate distance matrix for peptides
    PxQ = process.pairwise_distances(P, Q)  # calculate distance matrix for spectra-peptides
    PxD = process.pairwise_distances(P, D)
    
    # # Get maskes
    p_len, q_len = len(P), len(Q)
    PQ_mask, _ = get_masks(counts, p_len, q_len)
    PQ_mask = PQ_mask.to(device)

    # Mine hardest positives
    PxQ_masked = PxQ * PQ_mask
    PxQ_max = Q[PxQ_masked.max(1).indices]

    # # Mine hardest negatives
    PD = torch.cat((P, D))
    PxP.fill_diagonal_(float("inf"))
    PxPD = torch.cat((PxP, PxD), axis=1)
    PxPD_min = PD[PxPD.min(1).indices]              # nearest peptide for each peptide

    PQ_neg_mask = (1 - PQ_mask).to(device)
    PxQ_masked = PxQ * PQ_neg_mask
    PxQ_masked[PQ_neg_mask < 0.1] = float("inf")
    PxQ_min = Q[PxQ_masked.min(1).indices]          # nearest spectrum for each peptide

    # # Calculate the loss
    loss =  triplet_loss(P, PxQ_max, PxPD_min)      # peptide-peptide negatives
    loss += triplet_loss(P, PxQ_max, PxQ_min)       # peptide-spectrum negatives
    
    loss = loss / 2

    return loss, process.pairwise_distances(Q, PD)