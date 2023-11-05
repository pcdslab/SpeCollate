import numpy as np
import torch

from src.atlesconfig import config
from src.atlesutils import simulatespectra as sim


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        # dist = dist - torch.diag(dist.diag())
        dist.fill_diagonal_(0.0)
    dist[torch.isnan(dist)] = 0.0  # set all nan values to zero
    return torch.clamp(dist, 0.0, np.inf)


def process_fasta_in_batch(model, file_path, spectra_batch_size):
    f = open(file_path)
    lines = f.readlines()
    f.close()

    batch_size = config.get_config(section="ml", key="batch_size")

    masses = []
    spectra_out = []
    peps = []

    start = 0
    i = 0
    while start < len(lines):
        print("Batch: " + str(i))
        i += 1

        print("Generating spectra...")
        spectra, l_masses, l_peps = sim.fasta_to_spectra(lines, start, spectra_batch_size)
        masses.extend(l_masses)
        peps.extend(l_peps)
        start = start + spectra_batch_size

        with torch.no_grad():
            print("Converting to tensor...")
            # dtype=torch.float
            spectra = np.asarray(spectra)
            spectraTensor = torch.as_tensor(spectra, dtype=torch.float)[:, None, :]
            spectra_loader = torch.utils.data.DataLoader(dataset=spectraTensor, batch_size=batch_size, shuffle=False)

            print("Running the model...")
            spectra_out.extend(run_model(model, spectra_loader))

    return spectra_out, masses, peps


def run_model(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out = []
        for _, data in enumerate(loader):
            data = data.to(device)
            out.extend(model(data)[0].cpu().detach().numpy())
    print(len(out))
    return out
