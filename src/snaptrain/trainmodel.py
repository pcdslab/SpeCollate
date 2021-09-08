import random as rand
#from matplotlib import pyplot as plt
import torch
import torch.nn as nn

from src.snapconfig import config
from src.snaptrain import process

rand.seed(37)

train_accuracy = []
train_loss = []
test_accuracy = []
test_loss = []
snp_weight = config.get_config(section="ml", key="snp_weight")
ce_weight = config.get_config(section="ml", key="ce_weight")
mse_weight = config.get_config(section="ml", key="mse_weight")
divider = snp_weight + ce_weight# + mse_weight

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


def train(model, device, train_loader, triplet_loss, cross_entropy_loss, mse_loss, optimizer, epoch):
    model.train()
    batch_size = config.get_config(section='ml', key='batch_size')
    h = model.module.init_hidden(batch_size)

    accurate_labels = 0
    accurate_labels_ch = 0
    all_labels = 0
    snp_loss = ce_loss = 0
    l = 0
    # with progressbar.ProgressBar(max_value=len(train_loader)) as p_bar:
    for idx, data in enumerate(train_loader):
        h = tuple([e.data for e in h])
        q_len = len(data[0])
        p_len = len(data[1])
        d_len = len(data[2])
        if p_len > d_len:
            seq_len = config.get_config(section='ml', key='pep_seq_len') # + 111
            zero_pad = torch.zeros(p_len - d_len, seq_len, dtype=torch.long)
            data[2] = torch.cat((data[2], zero_pad), 0)
        data[0] = data[0].to(device) # spectra
        data[1] = data[1].to(device) # peptides
        data[2] = data[2].to(device) # decoys
        data[3] = data[3].to(device) # spectrum charges
        # data[4] = data[4].to(device) # spectrum masses
        # data[5] = data[5].to(device) # peptide masses
        # data[6] = data[6].to(device) # decoy masses
        counts = data[4]
        
        optimizer.zero_grad()
        
        # Q, ch, sm, P, pm, D, dm = model(data[:-1])
        Q, P, D = model(data[:-1])
        # assert len(Q) == len(ch)

        snp_loss, QxPD = snap_loss_2_d(counts, P, Q, D[:d_len], triplet_loss, device)
        
        # ce_loss = cross_entropy_loss(ch, data[3])
        # mse_loss_specs = mse_loss(sm, data[4])
        # mse_loss_peps = mse_loss(pm, data[5])
        # mse_loss_decs = mse_loss(dm[:d_len], data[6])
        # mse_tot_loss = mse_loss_specs + mse_loss_peps + mse_loss_decs
        # mse_tot_loss = mse_tot_loss / 3
        
        # loss = (snp_weight * snp_loss + ce_weight * ce_loss) / divider
        loss = snp_loss
        # print("snp loss: {}, ce loss: {}, mse loss: {}".format(snp_loss, ce_loss, mse_tot_loss))
        
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        optimizer.step()

        seq, _ = get_index(counts, q_len)
        seq = torch.LongTensor(seq).to(device)
        # QxP contains the distance of each spectrum from each peptide.

        accurate_labels += torch.sum(QxPD.argmin(1) == seq)
        # accurate_labels_ch += multi_acc(ch, data[3])
        all_labels += len(Q)
        # p_bar.update(idx)
    
    accuracy = 100. * float(accurate_labels) / all_labels
    # accuracy_ch = 100. * float(accurate_labels_ch) / all_labels

    # wandb.log({"train loss": snp_loss, "train accuracy": accuracy}, step=epoch)
    print('Train accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, snp_loss))
    # print('Train accuracy charge:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_ch, all_labels, accuracy_ch, ce_loss))
    # print('Train loss mse:\t{:.6f}'.format(mse_tot_loss))
    return loss
    

def test(model, device, test_loader, triplet_loss, cross_entropy_loss, mse_loss, epoch):
    model.eval()
    
    with torch.no_grad():
        accurate_labels = 0
        accurate_labels_ch = 0
        all_labels = 0
        snp_loss = ce_loss = 0

        batch_size = config.get_config(section='ml', key='batch_size')
        h = model.module.init_hidden(batch_size)
        
        for data in test_loader:
            h = tuple([e.data for e in h])
            q_len = len(data[0])
            p_len = len(data[1])
            d_len = len(data[2])
            if p_len > d_len:
                seq_len = config.get_config(section='ml', key='pep_seq_len') # + 111
                zero_pad = torch.zeros(p_len - d_len, seq_len, dtype=torch.long)
                data[2] = torch.cat((data[2], zero_pad))
            data[0] = data[0].to(device) # spectra
            data[1] = data[1].to(device) # peptides
            data[2] = data[2].to(device) # decoys
            data[3] = data[3].to(device) # spectrum charges
            # data[4] = data[4].to(device) # spectrum masses
            # data[5] = data[5].to(device) # peptide masses
            # data[6] = data[6].to(device) # decoy masses
            counts = data[4]
            
            # Q, ch, sm, P, pm, D, dm = model(data[:-1])
            Q, P, D = model(data[:-1])

            snp_loss, QxPD = snap_loss_2_d(counts, P, Q, D[:d_len], triplet_loss, device)
            # ce_loss = cross_entropy_loss(ch, data[3])
            # mse_loss_specs = mse_loss(sm, data[4])
            # mse_loss_peps = mse_loss(pm, data[5])
            # mse_loss_decs = mse_loss(dm[:d_len], data[6])
            # mse_tot_loss = mse_loss_specs + mse_loss_peps + mse_loss_decs
            # mse_tot_loss = mse_tot_loss / 3
            # loss = (snp_weight * snp_loss + ce_weight * ce_loss) / divider
            # loss = snp_loss
            
            seq, _ = get_index(counts, q_len)
            seq = torch.LongTensor(seq).to(device)
            # QxP contains the distance of each spectrum from each peptide.
            accurate_labels += torch.sum(QxPD.argmin(1) == seq)
            # accurate_labels_ch += multi_acc(ch, data[3])
            all_labels = all_labels + len(Q)
                
        accuracy = 100. * float(accurate_labels) / all_labels
        # accuracy_ch = 100. * float(accurate_labels_ch) / all_labels
        
        # wandb.log({"test loss": snp_loss, "test accuracy": accuracy}, step=epoch)
        print('Test accuracy:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels, all_labels, accuracy, snp_loss))
        # print('Test accuracy charge:\t{}/{} ({:.3f}%)\t\tLoss: {:.6f}'.format(accurate_labels_ch, all_labels, accuracy_ch, ce_loss))
        # print('Test loss mse:\t{:.6f}'.format(mse_tot_loss))


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
    assert len(rows) == len(cols)
    return rows, cols


def snap_loss(counts, Q, P, triplet_loss, device):
    """Mine the hardest sextuplets.""" 
    QxQ = process.pairwise_distances(Q)     # calculate distance matrix for spectra
    PxP = process.pairwise_distances(P)     # calculate distance matrix for peptides
    QxP = process.pairwise_distances(Q, P)  # calculate distance matrix for spectra-peptides
    
    # Get masks
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


def snap_loss_d(counts, P, Q, D, triplet_loss, device):
    """Mine the hardest sextuplets.""" 
    QxQ = process.pairwise_distances(Q)     # calculate distance matrix for spectra
    PxP = process.pairwise_distances(P)     # calculate distance matrix for peptides
    QxP = process.pairwise_distances(Q, P)  # calculate distance matrix for spectra-peptides
    QxD = process.pairwise_distances(Q, D)
    PxD = process.pairwise_distances(P, D)
    
    # Get masks
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
    QxQ_min = Q[QxQ_masked.min(1).indices]       # nearest spectrum for each spectrum

    PD = torch.cat((P, D))
    PxP.fill_diagonal_(float("inf"))
    PxPD = torch.cat((PxP, PxD), axis=1)
    PxPD_min = PD[PxPD.min(1).indices]           # nearest peptide for each peptide

    PQ_neg_mask = (1 - PQ_mask)
    PQ_neg_mask_clone = PQ_neg_mask.clone()
    PxQ_masked = torch.transpose(QxP, 0, 1) * PQ_neg_mask
    PxQ_masked[PQ_neg_mask_clone < 0.1] = float("inf")
    PxQ_min = Q[PxQ_masked.min(1).indices]       # nearest spectrum for each peptide

    QP_neg_mask = torch.transpose(PQ_neg_mask, 0, 1)
    QP_neg_mask_clone = QP_neg_mask.clone()
    QxP_masked = QxP * QP_neg_mask
    QxP_masked[QP_neg_mask_clone < 0.1] = float("inf")
    QxPD = torch.cat((QxP_masked, QxD), axis=1)
    QxPD_min = PD[QxPD.min(1).indices]            # nearest peptide for each spectrum

    # Calculate the loss
    loss  = triplet_loss(Q, QxP_max, QxQ_min)    # spectrum-spectrum negatives
    loss += triplet_loss(Q, QxP_max, QxPD_min)   # spectrum-peptide negatives
    loss += triplet_loss(P, PxQ_max, PxPD_min)   # peptide-peptide negatives
    loss += triplet_loss(P, PxQ_max, PxQ_min)    # peptide-spectrum negatives
    
    loss = loss / 4
    
    QxPD = process.pairwise_distances(Q, PD)
    qpd = torch.cat((QxP, QxD), axis=1)
    return loss, QxPD


def snap_loss_2(counts, P, Q, triplet_loss, device):
    """Mine the hardest sextuplets."""
    PxP = process.pairwise_distances(P)     # calculate distance matrix for peptides
    PxQ = process.pairwise_distances(P, Q)  # calculate distance matrix for spectra-peptides
    
    # # Get masks
    p_len, q_len = len(P), len(Q)
    PQ_mask, _ = get_masks(counts, p_len, q_len)
    PQ_mask = PQ_mask.to(device)

    # Mine hardest positives
    PxQ_masked = PxQ * PQ_mask
    PxQ_max = Q[PxQ_masked.max(1).indices]

    delta_plus = PxQ_masked.max(1).values #this one

    # # Mine hardest negatives
    PxP.fill_diagonal_(float("inf"))
    PxP_min = P[PxP.min(1).indices]              # nearest peptide for each peptide

    delta_minus_P = PxP.min(1).values # this one

    PQ_neg_mask = (1 - PQ_mask).to(device)
    PxQ_masked = PxQ * PQ_neg_mask
    PxQ_masked[PQ_neg_mask < 0.1] = float("inf")
    PxQ_min = Q[PxQ_masked.min(1).indices]          # nearest spectrum for each peptide
    
    delta_minus_Q = PxQ_masked.min(1).values # this one

    margin = config.get_config(section="ml", key="margin")
    filt1 = ((delta_plus - delta_minus_P)) > 0
    # if len(delta_plus) > torch.sum(filt1):
    #     print("filt1 length: {}".format(torch.sum(filt1).item()))
    A1 = P#[filt1]
    P1 = PxQ_max#[filt1]
    N1 = PxP_min#[filt1]

    filt2 = ((delta_plus - delta_minus_Q)) > 0
    # if len(delta_plus) > torch.sum(filt2):
    #     print("filt2 length: {}".format(torch.sum(filt2).item()))
    #     print("out of {}".format(len(delta_plus)))
    A2 = P#[filt2]
    P2 = PxQ_max#[filt2]
    N2 = PxQ_min#[filt2]

    # # Calculate the loss
    loss =  triplet_loss(A1, P1, N1)       # peptide-peptide negatives
    loss += triplet_loss(A2, P2, N2)       # peptide-spectrum negatives  
    loss = loss / 2

    return loss, process.pairwise_distances(Q, P)


def snap_loss_2_d(counts, P, Q, D, triplet_loss, device):
    """Mine the hardest sextuplets."""
    PxP = process.pairwise_distances(P)     # calculate distance matrix for peptides
    PxQ = process.pairwise_distances(P, Q)  # calculate distance matrix for spectra-peptides
    PxD = process.pairwise_distances(P, D)
    
    # # Get masks
    p_len, q_len = len(P), len(Q)
    PQ_mask, _ = get_masks(counts, p_len, q_len)
    PQ_mask = PQ_mask.to(device)

    # Mine hardest positives
    PxQ_masked = PxQ * PQ_mask
    PxQ_max = Q[PxQ_masked.max(1).indices]

    delta_plus = PxQ_masked.max(1).values #this one

    # # Mine hardest negatives
    PD = torch.cat((P, D))
    PxP.fill_diagonal_(float("inf"))
    PxPD = torch.cat((PxP, PxD), axis=1)
    PxPD_min = PD[PxPD.min(1).indices]              # nearest peptide for each peptide

    delta_minus_P = PxPD.min(1).values # this one

    PQ_neg_mask = (1 - PQ_mask).to(device)
    PxQ_masked = PxQ * PQ_neg_mask
    PxQ_masked[PQ_neg_mask < 0.1] = float("inf")
    PxQ_min = Q[PxQ_masked.min(1).indices]          # nearest spectrum for each peptide
    
    delta_minus_Q = PxQ_masked.min(1).values # this one

    margin = config.get_config(section="ml", key="margin")
    filt1 = ((delta_plus - delta_minus_P)) > 0
    # if len(delta_plus) > torch.sum(filt1):
    #     print("filt1 length: {}".format(torch.sum(filt1).item()))
    A1 = P#[filt1]
    P1 = PxQ_max#[filt1]
    N1 = PxPD_min#[filt1]

    filt2 = ((delta_plus - delta_minus_Q)) > 0
    # if len(delta_plus) > torch.sum(filt2):
    #     print("filt2 length: {}".format(torch.sum(filt2).item()))
    #     print("out of {}".format(len(delta_plus)))
    A2 = P#[filt2]
    P2 = PxQ_max#[filt2]
    N2 = PxQ_min#[filt2]

    # # Calculate the loss
    loss =  triplet_loss(A1, P1, N1)       # peptide-peptide negatives
    loss += triplet_loss(A2, P2, N2)       # peptide-spectrum negatives  
    loss = loss / 2

    return loss, process.pairwise_distances(Q, PD)


# TODO: change it. taken from 
# https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab
# accessed: 09/18/2020
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float().sum()
    
    return correct_pred