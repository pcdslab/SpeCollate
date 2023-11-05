import pickle
from bisect import bisect_left, bisect_right

import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm

def find_indices_within_tolerance(precursor_masses, target_mass, tol):
    lower_mass = target_mass - tol
    upper_mass = target_mass + tol

    lower_index = bisect_left(precursor_masses, lower_mass)
    upper_index = bisect_right(precursor_masses, upper_mass)

    return lower_index, upper_index

def calculate_smallest_k_diffs(precursors1, precursors2, tol, k):
    # Initialize results matrix as a sparse matrix
    precursor_diff_matrix = lil_matrix((len(precursors1), len(precursors2)), dtype=np.float32)

    # Iterate over all precursor masses in precursors1
    for i in tqdm(range(len(precursors1)), desc="Calculating precursor differences"):
        # Find precursor masses in precursors2 within tolerance
        start_index = bisect_left(precursors2, precursors1[i] - tol)
        end_index = bisect_right(precursors2, precursors1[i] + tol)

        for j in range(start_index, end_index):
            # Calculate precursor difference
            precursor_diff_matrix[i, j] = np.abs(precursors1[i] - precursors2[j])

    # Convert to CSR format for efficient arithmetic and matrix operations
    precursor_diff_matrix = precursor_diff_matrix.tocsr()

    # Initialize an array to store the smallest k differences for each row
    smallest_k_diffs = np.zeros((precursor_diff_matrix.shape[0], k))

    # For each row in the precursor difference matrix
    for i in range(precursor_diff_matrix.shape[0]):
        # Convert the row to a dense format and flatten to a 1D array
        row_diffs = precursor_diff_matrix[i].toarray().flatten()

        # Find the indices of the k smallest differences
        smallest_k_indices = np.argpartition(row_diffs, k)[:k]

        # Extract the k smallest differences and store in the smallest_k_diffs array
        smallest_k_diffs[i] = row_diffs[smallest_k_indices]

    return smallest_k_diffs


if __name__ == "__main__":
    k = 1
    print("Loading data...")
    test_e_specs = np.load("uncertainty_analysis/proteome_tools_data/cid/e_specs.npy")
    test_np_specs = pickle.load(open("uncertainty_analysis/proteome_tools_data/cid/np_specs.pkl", "rb"))
    test_spec_masses = np.load("uncertainty_analysis/proteome_tools_data/cid/spec_masses.npy")

    q = np.load("uncertainty_analysis/training_data/q.npy")
    train_np_specs = pickle.load(open("uncertainty_analysis/training_data/np_specs.pkl", "rb"))
    train_spec_masses = np.load("uncertainty_analysis/training_data/masses.npy").flatten()

    precursor_diff = calculate_smallest_k_diffs(test_spec_masses[::k], train_spec_masses, tol=10, k=5)

    np.save("uncertainty_analysis/outputs/precursor_diff_ood.npy", precursor_diff)
