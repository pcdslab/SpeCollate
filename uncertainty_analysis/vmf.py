import numpy as np
from scipy.special import iv, logsumexp
from spherecluster import VonMisesFisherMixture


def fit_vmf_and_estimate_density_2(data, p, n_clusters=3, max_iter=100, tol=1e-3):
    m = p.shape[1]

    # Fit a von Mises-Fisher Mixture Model to the training data p
    vmf = VonMisesFisherMixture(n_clusters=n_clusters, max_iter=max_iter, tol=tol)
    print("Fitting VMF model...")
    vmf.fit(p)

    # Compute the log of the density estimate for each data point
    print("Computing density estimate...")
    log_exp_terms = np.dot(data, vmf.cluster_centers_.T) * vmf.concentrations_
    log_const_terms = (np.log(vmf.concentrations_) * ((m - 1) / 2)) - (
        np.log(2 * np.pi) * (m / 2) + np.log(iv((m - 1) / 2, vmf.concentrations_))
    )
    log_density_estimate = logsumexp(log_const_terms + log_exp_terms, axis=1, b=vmf.weights_)

    # Normalize the log density estimates and convert back to the density estimate
    max_log_density_estimate = np.max(log_density_estimate)
    shifted_log_density_estimate = log_density_estimate - max_log_density_estimate
    sum_exp_shifted_log_density_estimate = np.sum(np.exp(shifted_log_density_estimate))
    normalized_log_density_estimate = shifted_log_density_estimate - np.log(sum_exp_shifted_log_density_estimate)
    density_estimate = np.exp(normalized_log_density_estimate)

    return log_density_estimate


def normalize_values(values):
    min_val, max_val = min(values), max(values)
    vals = np.array(values)
    normalized_values = (vals - min_val) / (max_val - min_val)
    print(min_val, max_val)
    print(normalized_values)
    return normalized_values


if __name__ == "__main__":
    # Load the data
    k = 1
    frag_type = "hcd" # cid or hcd
    save_append = "_rmv_peaks_augmnts"
    test_e_specs = np.load(f"uncertainty_analysis/proteome_tools_data/{frag_type}/e_specs{save_append}.npy")
    p = np.load("uncertainty_analysis/training_data/p.npy")

    print("test_e_specs.shape: ", test_e_specs.shape)
    print("p.shape: ", p.shape)

    prob_density_specs = fit_vmf_and_estimate_density_2(test_e_specs[::k], p, n_clusters=8)

    # convert prob_density_specs to numpy array save it to "uncertainty_analysis/outputs/prob_density_specs.npy"
    prob_density_specs = np.array(prob_density_specs)
    np.save(f"uncertainty_analysis/outputs/prob_density_specs{save_append}.npy", prob_density_specs)
