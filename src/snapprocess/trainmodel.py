import random as rand

import torch
import torch.nn as nn

from src.snapconfig import config
from src.snapprocess import process

rand.seed(37)

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
        h = tuple([e.data for e in h])
        data[0], data[1] = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        Q, P, h = model(data, h)

        """Mine the hardest triplets. Get rid of N.""" 
        QxQ = process.pairwise_distances(Q)    # calculate distance matrix for spectra
        PxP = process.pairwise_distances(P)    # calculate distance matrix for peptides
        QxP_ = process.pairwise_distances(Q, P) # calculate distance matrix for spectra-peptides
        
        # Set the diagonal of all distance matrices to inf so we don't get self as the closest negative.
        QxQ.fill_diagonal_(float("inf"))
        PxP.fill_diagonal_(float("inf"))
        QxP = QxP_.clone()
        QxP.fill_diagonal_(float("inf"))
        
        #print(QP.argmin(1)[:100])
        
#         pos = torch.sum(l2_squared(Q, P), dim=1) + margin
        
#         QxQ_min = QxQ.gather(1, torch.randint(len(Q), (len(Q),), device=device).view(-1,1))             # farthest spectrum for each spectrum
#         PxP_min = PxP.gather(1, torch.randint(len(Q), (len(Q),), device=device).view(-1,1))             # farthest peptide for each peptide
#         QxP_min = QxP.gather(1, torch.randint(len(Q), (len(Q),), device=device).view(-1,1))             # farthest peptide for each spectrum
#         PxQ_min = QxP.gather(0, torch.randint(len(Q), (len(Q),), device=device).view(1,-1))             # farthest spectrum for each peptide
        
        QxQ_min = QxQ.min(1).values              # nearest spectrum for each spectrum
        PxP_min = PxP.min(1).values              # nearest peptide for each peptide
        QxP_min = QxP.min(1).values              # nearest peptide for each spectrum
        PxQ_min = QxP.min(0).values              # nearest spectrum for each peptide
        
        #neg = QxQ_min + PxP_min + QxP_min + PxQ_min
        
#         divider = torch.tensor(float(len(pos)))
#         loss = torch.sum(torch.max(pos - QxQ_min, zero_tensor)) / divider
#         loss += torch.sum(torch.max(pos - PxP_min, zero_tensor)) / divider
#         loss += torch.sum(torch.max(pos - QxP_min, zero_tensor)) / divider
#         loss += torch.sum(torch.max(pos - PxQ_min, zero_tensor)) / divider
        
        #divider = torch.sum(pos - neg > 0)
        #loss = torch.sum(torch.max(pos - neg, zero_tensor)) / divider
        
        QxQ_min = Q[QxQ.min(1).indices]              # nearest spectrum for each spectrum
        PxP_min = P[PxP.min(1).indices]              # nearest peptide for each peptide
        QxP_min = P[QxP.min(1).indices]              # nearest peptide for each spectrum
        PxQ_min = Q[QxP.min(0).indices]              # nearest spectrum for each peptide
        loss = triplet_loss(Q, P, QxQ_min)           # spectrum-spectrum negatives
        loss += triplet_loss(Q, P, QxP_min)          # spectrum-peptide negatives
        loss += triplet_loss(P, Q, PxP_min)          # peptide-peptide negatives
        loss += triplet_loss(P, Q, PxQ_min)          # peptide-spectrum negatives
        
        loss = loss / 4
                
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        seq = torch.arange(0, len(Q), step=1, device=device, requires_grad=False)
        accurate_labels = accurate_labels + torch.sum(QxP_.argmin(1) == seq) # use QP_ since it doesn't have diag set to zero
        
        all_labels = all_labels + len(Q)  
    
    accuracy = 100. * float(accurate_labels) / all_labels
    # train_accuracy.append(accuracy)
    # train_loss.append(loss)
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
            
            Q, P, h = model(data, h)
            
            """Mine the hardest triplets. Get rid of N."""
            QxQ = process.pairwise_distances(Q)     # calculate distance matrix for spectra
            PxP = process.pairwise_distances(P)     # calculate distance matrix for peptides
            QxP_ = process.pairwise_distances(Q, P) # calculate distance matrix for spectra-peptides

            # Set the diagonal of all distance matrices to inf so we don't get self as the closest negative.
            QxQ.fill_diagonal_(float("inf"))
            PxP.fill_diagonal_(float("inf"))
            QxP = QxP_.clone()    # clone to measure accuracy. can be done in a better way.
            QxP.fill_diagonal_(float("inf"))

            #print(QP.argmin(1)[:100])

            QxQ_min = QxQ.min(1).values              # farthest spectrum for each spectrum
            PxP_min = PxP.min(1).values              # farthest peptide for each peptide
            QxP_min = QxP.min(1).values              # farthest peptide for each spectrum
            PxQ_min = QxP.min(0).values              # farthest spectrum for each peptide

            #neg = QxQ_min + PxP_min + QxP_min + PxQ_min
        
#             divider = torch.tensor(float(len(pos)))
#             loss = torch.sum(torch.max(pos - QxQ_min, zero_tensor)) / divider
#             loss += torch.sum(torch.max(pos - PxP_min, zero_tensor)) / divider
#             loss += torch.sum(torch.max(pos - QxP_min, zero_tensor)) / divider
#             loss += torch.sum(torch.max(pos - PxQ_min, zero_tensor)) / divider

            QxQ_min = Q[QxQ.min(1).indices]              # nearest spectrum for each spectrum
            PxP_min = P[PxP.min(1).indices]              # nearest peptide for each peptide
            QxP_min = P[QxP.min(1).indices]              # nearest peptide for each spectrum
            PxQ_min = Q[QxP.min(0).indices]              # nearest spectrum for each peptide
            loss = triplet_loss(Q, P, QxQ_min)     # spectrum-spectrum negatives
            loss += triplet_loss(Q, P, QxP_min)    # spectrum-peptide negatives
            loss += triplet_loss(P, Q, PxP_min)    # peptide-peptide negatives
            loss += triplet_loss(P, Q, PxQ_min)    # peptide-spectrum negatives
            
            loss = loss / 4

            #divider = torch.tensor(float(len(pos)))
            #divider = torch.sum(pos - neg > 0)
            #loss = torch.sum(torch.max(pos - neg, zero_tensor)) / divider
            
#             loss =  torch.sum(torch.max(pos - QxQ_min, zero_tensor)) / divider
#             loss += torch.sum(torch.max(pos - PxP_min, zero_tensor)) / divider
#             loss += torch.sum(torch.max(pos - QxP_min, zero_tensor)) / divider
#             loss += torch.sum(torch.max(pos - PxQ_min, zero_tensor)) / divider
            
            seq = torch.arange(0, len(Q), step=1, device=device, requires_grad=False)
            accurate_labels = accurate_labels + torch.sum(QxP_.argmin(1) == seq) # use QP_ since it doesn't have diag set to zero
            
            all_labels = all_labels + len(Q)
                
        accuracy = 100. * float(accurate_labels) / all_labels
        # test_accuracy.append(accuracy)
        # test_loss.append(loss)
        print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, loss))