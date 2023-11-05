import pickle
from bisect import bisect_left, bisect_right

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
import cProfile
from tqdm import tqdm


def find_indices_within_tolerance(precursor_masses, target_mass, tol):
    lower_mass = target_mass - tol
    upper_mass = target_mass + tol

    lower_index = bisect_left(precursor_masses, lower_mass)
    upper_index = bisect_right(precursor_masses, upper_mass)

    return lower_index, upper_index


def spectral_similarity(spec1, spec2, norm1, norm2):
    mz1 = spec1[0].astype(int)
    mz2 = spec2[0].astype(int)

    intensity1 = spec1[1]
    intensity2 = spec2[1]

    dot_product = 0
    i, j = 0, 0
    while i < len(mz1) and j < len(mz2):
        if mz1[i] == mz2[j]:
            dot_product += intensity1[i] * intensity2[j]
            i += 1
            j += 1
        elif mz1[i] < mz2[j]:
            i += 1
        else:
            j += 1

    if norm1 != 0 and norm2 != 0:
        similarity = dot_product / (norm1 * norm2)
    else:
        similarity = 0

    return similarity


def msms_similarity(specs1, specs2, emb1, emb2, precursors1, precursors2, tol):
    # Initialize results matrix as a sparse matrix
    spec_sim_matrix = lil_matrix((len(specs1), len(specs2)), dtype=np.float32)
    precursor_diff_matrix = lil_matrix((len(specs1), len(specs2)), dtype=np.float32)
    emb_sim_matrix = lil_matrix((len(emb1), len(emb2)), dtype=np.float32)

    # Precompute the intensity norms for each spectrum for speed
    specs1_intensity_norms = [np.linalg.norm(spec[1]) for spec in specs1]
    specs2_intensity_norms = [np.linalg.norm(spec[1]) for spec in specs2]

    # Iterate over all spectra in specs1
    for i in tqdm(range(len(specs1)), desc="Calculating spectral similarity and precursor differences"):
        # Find spectra in specs2 within tolerance
        start_index = bisect_left(precursors2, precursors1[i] - tol)
        end_index = bisect_right(precursors2, precursors1[i] + tol)

        for j in range(start_index, end_index):
            # Calculate spectral similarity
            spec_sim_matrix[i, j] = spectral_similarity(specs1[i], specs2[j], specs1_intensity_norms[i], specs2_intensity_norms[j])

            # Calculate precursor difference
            precursor_diff_matrix[i, j] = np.abs(precursors1[i] - precursors2[j])

            # Calculate embedding similarity
            emb_sim_matrix[i, j] = cosine_similarity(emb1[i].reshape(1, -1), emb2[j].reshape(1, -1))[0][0]

    return spec_sim_matrix.tocsr(), precursor_diff_matrix.tocsr(), emb_sim_matrix.tocsr()


def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return (values - min_val) / (max_val - min_val)


def embedding_confidence(test_X, train_X, test_Y, train_Y, test_precursors, train_precursors, tol, k):
    num_test = len(test_X)
    l_consistency = np.zeros(num_test)
    l_sim_X = np.zeros(num_test)
    l_precursor_diff = np.zeros(num_test)
    l_sim_Y = np.zeros(num_test)

    # Get the spectral similarity, precursor differences, and embedding similarity matrices
    spec_sim_matrix, precursor_diff_matrix, emb_sim_matrix = msms_similarity(
        test_X, train_X, test_Y, train_Y, test_precursors, train_precursors, tol
    )

    for i in tqdm(range(num_test), desc="Get top k and calculate confidence"):
        # Get the spectral similarities, precursor differences, and embedding similarities for the i-th test example
        spec_similarities = spec_sim_matrix[i].toarray().flatten()
        precursor_diffs = precursor_diff_matrix[i].toarray().flatten()
        emb_similarities = emb_sim_matrix[i].toarray().flatten()

        # Get the indices of the top-k most similar examples
        top_k_spec_indices = np.argpartition(spec_similarities, -k)[-k:]
        top_k_precursor_diff_indices = np.argpartition(precursor_diffs, k)[:k]
        top_k_emb_indices = np.argpartition(emb_similarities, -k)[-k:]

        # Calculate the mean similarities/differences in the input and embedding spaces
        mean_sim_X = np.mean(spec_similarities[top_k_spec_indices])
        mean_diff_precursor = np.mean(precursor_diffs[top_k_precursor_diff_indices])
        mean_sim_Y = np.mean(emb_similarities[top_k_emb_indices])

        # Calculate the confidence score for the i-th test example
        l_sim_X[i] = mean_sim_X
        l_precursor_diff[i] = mean_diff_precursor
        l_sim_Y[i] = mean_sim_Y
        l_consistency[i] = np.abs(mean_sim_X - mean_sim_Y)

    return l_consistency, l_sim_X, l_precursor_diff, l_sim_Y


def main():
    k = 1
    frag_type = "hcd" # cid or hcd
    save_append = "_rmv_peaks_augmnts"
    print("Loading data...")
    test_e_specs = np.load(f"uncertainty_analysis/proteome_tools_data/{frag_type}/e_specs{save_append}.npy")
    test_np_specs = pickle.load(open(f"uncertainty_analysis/proteome_tools_data/{frag_type}/np_specs{save_append}.pkl", "rb"))
    test_spec_masses = np.load(f"uncertainty_analysis/proteome_tools_data/{frag_type}/spec_masses{save_append}.npy")

    q = np.load("uncertainty_analysis/training_data/q.npy") # q := train_e_specs
    train_np_specs = pickle.load(open("uncertainty_analysis/training_data/np_specs.pkl", "rb"))
    train_spec_masses = np.load("uncertainty_analysis/training_data/masses.npy").flatten()

    consistency, sim_X, precursor_diff, sim_Y = embedding_confidence(
        test_np_specs[::k], train_np_specs, test_e_specs[::k], q, test_spec_masses[::k], train_spec_masses, tol=10, k=5
    )

    # Save the confidence scores
    np.save(f"uncertainty_analysis/outputs/consistency{save_append}.npy", consistency)
    np.save(f"uncertainty_analysis/outputs/sim_X{save_append}.npy", sim_X)
    np.save(f"uncertainty_analysis/outputs/precursor_diff{save_append}.npy", precursor_diff)
    np.save(f"uncertainty_analysis/outputs/sim_Y{save_append}.npy", sim_Y)


if __name__ == "__main__":
    cProfile.run("main()", "uncertainty_analysis/outputs/embedding_confidence.prof")
