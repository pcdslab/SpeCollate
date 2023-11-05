import numpy as np
from scipy.special import iv, logsumexp
from spherecluster import VonMisesFisherMixture

p = np.load("uncertainty_analysis/training_data/p.npy")
vmf = VonMisesFisherMixture(n_clusters=8, max_iter=64, tol=1e-3)
print("Fitting VMF model...")
vmf.fit(p)

# save the model
np.save("uncertainty_analysis/models/vmf/vmf_cluster_centers.npy", vmf.cluster_centers_)

# save the weights
np.save("uncertainty_analysis/models/vmf/vmf_weights.npy", vmf.weights_)

# save the concentrations
np.save("uncertainty_analysis/models/vmf/vmf_concentrations.npy", vmf.concentrations_)

# load the model
# vmf_cluster_centers = np.load("uncertainty_analysis/models/vmf/vmf_cluster_centers.npy")
# vmf_weights = np.load("uncertainty_analysis/models/vmf/vmf_weights.npy")
# vmf_concentrations = np.load("uncertainty_analysis/models/vmf/vmf_concentrations.npy")
