import random as rand
import atexit
#from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from src.snapconfig import config
from src.snapprocess import process

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
    for data in train_loader:
        #print("l: {}/{}".format(l, len(train_loader)))
        #l += 1
        #print(len(data))
        h = tuple([e.data for e in h])
        # print(type(data[0][0]))
        # print((data[0][0].shape))
        # print(type(data[1]))
        # print(data[1].shape)
        # print(type(data[2]))
        # print((data[2].shape))
        data[0], data[1] = data[0].to(device), data[1].to(device)
        counts = data[2]
        q_len = len(data[0])
        
        optimizer.zero_grad()
        
        Q, P, h = model(data, h)

        loss, QxP = snap_loss(counts, Q, P, triplet_loss, device)
                
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()
        
        seq, _ = get_index(counts, q_len)
        seq = torch.LongTensor(seq).to(device)
        #print("accuracy seq len: {}".format(len(seq)))
        #print("QxP.argmin(1) len: {}".format(len(QxP.argmin(1))))
        # QxP contains the distance of each spectrum from each peptide.
        accurate_labels = accurate_labels + torch.sum(QxP.argmin(1) == seq)
        
        all_labels = all_labels + len(Q)  
    
    accuracy = 100. * float(accurate_labels) / all_labels
    train_accuracy.append(accuracy)
    train_loss.append(loss)
    print('Train accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))
    

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
            data[0], data[1] = data[0].to(device), data[1].to(device)
            counts = data[2]
            q_len = len(data[0])
            
            Q, P, h = model(data, h)

            loss, QxP = snap_loss(counts, Q, P, triplet_loss, device)
            
            seq, _ = get_index(counts, q_len)
            seq = torch.LongTensor(seq).to(device)
            # QxP contains the distance of each spectrum from each peptide.
            accurate_labels = accurate_labels + torch.sum(QxP.argmin(1) == seq)
            
            all_labels = all_labels + len(Q)
                
        accuracy = 100. * float(accurate_labels) / all_labels
        test_accuracy.append(accuracy)
        test_loss.append(loss)
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))


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