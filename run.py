# --- Imports ---
# Standard libraries
from itertools import product
import random
import os

# Local
import algorithm_functions as af
import helper_functions as hp

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# --- Global variables ---
zero_threshold = 0.00001

# --- Helper functions ---
def calculate_diffs_new(pair, nlatent, nobserved):
    """
    Calculate the errors when recovery is performed using the general method.

    Args:
        pair (tuple): tuple representing the true model. It contains the true mixing matrix, a dictionary mapping each context (str) to the matrix representing its corresponding latent graph, and a dictionary mapping each context to the true product that should be recovered from tensor decomposition.
        nlatent (int): number of latent variables.
        nobserved (int): number of observed variables.
    Returns:
        A tuple containing the relative errors in estimating F, Lambda, and the latent graph, both directly from the matrices and from the cumulants.
    """
    F, Lambdas, Products_og = pair
    _, realTs, Products, _ = af.construct_cumulants_kstat(4, Lambdas, F, 4)
    RProducts = af.decompose_tensors(realTs, nobserved, nlatent, 4)
    perm = hp.find_best_permutation(Products["obs"].copy(),RProducts["obs"].copy())

    aligned_prods = af.align_outputs(RProducts, True)
    _, Lambdar = af.recover_lambda(aligned_prods,zero_threshold)
    Fr = af.recover_F(aligned_prods["obs"][0],Lambdar)

    lambda_error = hp.lambda_diff(Lambdas["obs"], Lambdar, perm)
    graph_error = hp.graph_diff(Lambdas["obs"], Lambdar, perm)
    f_error = hp.F_diff(F,Fr,perm, True)

    # Recover directly from matrices
    Products_matrix = {k: v[0] for k,v in Products_og.items()}
    perm_matrix = np.eye(nlatent)
    np.random.shuffle(perm_matrix)
    perm_matrix_obs = perm_matrix.copy()
    scaling_list = list(product([1,-1], repeat=nlatent))
    scale_obs = random.choice(scaling_list)
    Products_matrix["obs"] = np.matmul(np.matmul(Products_matrix["obs"],perm_matrix_obs),np.diag(scale_obs))
    for i in range(nlatent):
        scale = random.choice(scaling_list)
        np.random.shuffle(perm_matrix)
        Products_matrix["{0}".format(i)] = np.matmul(np.matmul(Products_matrix["{0}".format(i)],perm_matrix),np.diag(scale))
    aligned_prods_matrix = af.align_outputs(Products_matrix, True)
    _, Lambdar_matrix = af.recover_lambda(aligned_prods_matrix, zero_threshold)
    lambda_error_matrix = hp.lambda_diff(Lambdas["obs"], Lambdar_matrix, perm_matrix_obs)
    graph_error_matrix = hp.graph_diff(Lambdas["obs"], Lambdar_matrix, perm_matrix_obs)
    Fr_matrix = af.recover_F(Products_matrix["obs"],Lambdar_matrix)
    f_error_matrix = hp.F_diff(F, Fr_matrix, perm_matrix_obs, True)
    return f_error, lambda_error, graph_error, f_error_matrix, lambda_error_matrix, graph_error_matrix

def calculate_diffs_old(pair, nlatent, nobserved):
    """
    Calculate the errors when recovery is performed using the injective method.

    Args:
        pair (tuple): tuple representing the true model. It contains the true mixing matrix, a dictionary mapping each context (str) to the matrix representing its corresponding latent graph, and a dictionary mapping each context to the true product that should be recovered from tensor decomposition.
        nlatent (int): number of latent variables.
        nobserved (int): number of observed variables.
    Returns:
        A tuple containing the relative errors in estimating F, Lambda, and the latent graph, both directly from the matrices and from the cumulants.
    """
    F, Lambdas, Products_og = pair
                
    _, realTs, Products, _ = af.construct_cumulants_kstat(5,Lambdas,F)

    RProducts = af.decompose_tensors(realTs, nobserved, nlatent)
    RProdnninv = af.nonnorm_products(realTs, RProducts)
    RProdnn = {key: values[0] for key, values in RProdnninv.items()}
    int_tuples = af.match_int_injective(RProdnninv)

    perm = hp.find_best_permutation(Products["obs"],RProdnn["obs"])
    tuples_with_D = af.recover_Dprime(int_tuples)

    Hr = af.recover_H(tuples_with_D)
    Fr = np.linalg.pinv(Hr)

    diff_F = hp.F_diff(F,Fr,perm)
    Lambdar = af.recover_lambda_injective(tuples_with_D["obs"][0],Hr, zero_threshold)
    diff_lambda = hp.lambda_diff(Lambdas["obs"],Lambdar,perm)

    graph_error = hp.graph_diff(Lambdas["obs"], Lambdar, perm)

    # Recover directly from matrices
    int_tuples_matrix = af.match_int_injective(Products_og)

    tuples_with_D_matrix = af.recover_Dprime(int_tuples_matrix)

    perm_matrix = np.eye(nlatent)
    Hr_matrix = af.recover_H(tuples_with_D_matrix)
    Fr_matrix = np.linalg.pinv(Hr_matrix)

    diff_F_matrix = hp.F_diff(F,Fr_matrix,perm_matrix)
    Lambdar_matrix = af.recover_lambda_injective(tuples_with_D_matrix["obs"][0],Hr_matrix, zero_threshold)
    diff_lambda_matrix = hp.lambda_diff(Lambdas["obs"],Lambdar_matrix,perm_matrix)

    graph_error_matrix = hp.graph_diff(Lambdas["obs"], Lambdar_matrix, perm_matrix)

    return diff_F, diff_lambda, graph_error, diff_F_matrix, diff_lambda_matrix, graph_error_matrix

# Run experiment
def run():
    """
    Test performance of method for linear causal disentanglement, using both the general and the injective method and starting both from the population cumulants and directly from the products that should be recovered from tensor decomposition on these cumulants.

    Returns:
        A tuple containing error lists. Each list is characterized by the paremeter estimated (F, Lambda, or the graph), the method used to estimate it (general or injective), and the starting point for the recovery (cumulants or matrices). The element at index i in a list estimating parameter p is a tuple containing the mean and median errors incurred when estimating p across 500 random models (using the corresponding estimation method and starting point).
    """
    nobserved = 5
    nmodels = 500
    diff_F_global_new = []
    diff_lambda_global_new = []
    diff_graph_global_new = []
    diff_F_global_new_matrix = []
    diff_lambda_global_new_matrix = []
    diff_graph_global_new_matrix = []

    diff_F_global_old = []
    diff_lambda_global_old = []
    diff_graph_global_old = []
    diff_F_global_old_matrix = []
    diff_lambda_global_old_matrix = []
    diff_graph_global_old_matrix = []

    for nlatent in range(2, 8):
        print("q = {0}".format(nlatent))
        models = af.create_models(nmodels, nobserved, nlatent)
        diff_F_local_new = []
        diff_lambda_local_new = []
        diff_graph_local_new = []
        diff_F_local_new_matrix = []
        diff_lambda_local_new_matrix = []
        diff_graph_local_new_matrix = []

        if nlatent <= nobserved:
            diff_F_local_old = []
            diff_lambda_local_old = []
            diff_graph_local_old = []
            diff_F_local_old_matrix = []
            diff_lambda_local_old_matrix = []
            diff_graph_local_old_matrix = []

        for pair in models:
            f_error_new, lambda_error_new, graph_error_new, f_error_new_matrix, lambda_error_new_matrix, graph_error_new_matrix  = calculate_diffs_new(pair, nlatent,nobserved)
            diff_F_local_new.append(f_error_new)
            diff_lambda_local_new.append(lambda_error_new)
            diff_graph_local_new.append(graph_error_new)
            diff_F_local_new_matrix.append(f_error_new_matrix)
            diff_lambda_local_new_matrix.append(lambda_error_new_matrix)
            diff_graph_local_new_matrix.append(graph_error_new_matrix)

            if nlatent <= nobserved:
                f_error_old, lambda_error_old, graph_error_old, f_error_old_matrix, lambda_error_old_matrix, graph_error_old_matrix  = calculate_diffs_old(pair, nlatent,nobserved)
                diff_F_local_old.append(f_error_old)
                diff_lambda_local_old.append(lambda_error_old)
                diff_graph_local_old.append(graph_error_old)
                diff_F_local_old_matrix.append(f_error_old_matrix)
                diff_lambda_local_old_matrix.append(lambda_error_old_matrix)
                diff_graph_local_old_matrix.append(graph_error_old_matrix)

        print("Median error in F for q = {0} is {1} (SPM)".format(nlatent, np.median(diff_F_local_new)))
        diff_F_global_new.append((np.median(diff_F_local_new),np.mean(diff_F_local_new)))
        print("Median error in Lambda for q = {0} is {1} (SPM)".format(nlatent, np.median(diff_lambda_local_new)))
        diff_lambda_global_new.append((np.median(diff_lambda_local_new),np.mean(diff_lambda_local_new)))
        print("Median error in graph for q = {0} is {1} (SPM)".format(nlatent, np.median(diff_graph_local_new)))
        diff_graph_global_new.append((np.median(diff_graph_local_new),np.mean(diff_graph_local_new)))

        print("Median error in F for q = {0} is {1} (new - matrix)".format(nlatent, np.median(diff_F_local_new_matrix)))
        diff_F_global_new_matrix.append((np.median(diff_F_local_new_matrix),np.mean(diff_F_local_new_matrix)))
        print("Median error in Lambda for q = {0} is {1} (new - matrix)".format(nlatent, np.median(diff_lambda_local_new_matrix)))
        diff_lambda_global_new_matrix.append((np.median(diff_lambda_local_new_matrix),np.mean(diff_lambda_local_new_matrix)))
        print("Median error in graph for q = {0} is {1} (new - matrix)".format(nlatent, np.median(diff_graph_local_new_matrix)))
        diff_graph_global_new_matrix.append((np.median(diff_graph_local_new_matrix), np.mean(diff_graph_local_new_matrix)))

        diff_F_global_old.append((np.median(diff_F_local_old),np.mean(diff_F_local_old)) if nlatent<= nobserved else (None,None))
        diff_lambda_global_old.append((np.median(diff_lambda_local_old),np.mean(diff_lambda_local_old)) if nlatent<= nobserved else (None, None))
        diff_graph_global_old.append((np.median(diff_graph_local_old), np.mean(diff_graph_local_old)) if nlatent <= nobserved else (None,None))

        diff_F_global_old_matrix.append((np.median(diff_F_local_old_matrix),np.mean(diff_F_local_old_matrix)) if nlatent<= nobserved else (None,None))
        diff_lambda_global_old_matrix.append((np.median(diff_lambda_local_old_matrix),np.mean(diff_lambda_local_old_matrix)) if nlatent<= nobserved else (None,None))
        diff_graph_global_old_matrix.append((np.median(diff_graph_local_old_matrix),np.mean(diff_graph_local_old_matrix)) if nlatent <= nobserved else (None,None))
        if nlatent <= nobserved:
            print("Median error in F for q = {0} is {1} (old - matrix)".format(nlatent, np.median(diff_F_local_old_matrix)))
            print("Median error in Lambda for q = {0} is {1} (old - matrix)".format(nlatent, np.median(diff_lambda_local_old_matrix)))
            print("Median error in graph for q = {0} is {1} (old - matrix)".format(nlatent, np.median(diff_graph_local_old_matrix)))
            print("Median error in F for q = {0} is {1} (old)".format(nlatent, np.median(diff_F_local_old)))
            print("Median error in Lambda for q = {0} is {1} (old)".format(nlatent, np.median(diff_lambda_local_old)))
            print("Median error in graph for q = {0} is {1} (old)".format(nlatent, np.median(diff_graph_local_old)))
        
    return diff_F_global_new, diff_F_global_new_matrix, diff_F_global_old, diff_F_global_old_matrix, diff_lambda_global_new, diff_lambda_global_new_matrix, diff_lambda_global_old, diff_lambda_global_old_matrix, diff_graph_global_new, diff_graph_global_new_matrix, diff_graph_global_old, diff_graph_global_old_matrix

def plot(diff_F_global_new, diff_F_global_new_matrix, diff_F_global_old, diff_F_global_old_matrix, diff_lambda_global_new, diff_lambda_global_new_matrix,  diff_lambda_global_old , diff_lambda_global_old_matrix,diff_graph_global_new,  diff_graph_global_new_matrix, diff_graph_global_old, diff_graph_global_old_matrix):
    """
    Plot the results of testing the method for linear causal disentanglement and save the figures.

    Args:
        diff_F_global_new (list[(float, float)]): element at index i contains the mean and median errors (in that order) incurred when estimating F using the general method and starting from the population cumulants, across 500 models with q = i and p = 5.
        diff_F_global_new_matrix (list[(float, float)]): element at index i contains the mean and median errors (in that order) incurred when estimating F using the general method and starting from the matrices that should be obtained from tensor decomposition on the population cumulants, across 500 models with q = i and p = 5.
        diff_F_global_old (list[(float, float)]): if i <= 5, element at index i contains the mean and median errors (in that order) incurred when estimating F using the injective method and starting from the population cumulants, across 500 models with q = i and p = 5. If i > 5, element at index i is (None, None).
        diff_F_global_old_matrix (list[float, float]): if i <= 5, element at index i contains the mean and median errors (in that order) incurred when estimating F using the injective method and starting from the matrices that should be obtained from tensor decomposition on the population cumulants, across 500 models with q = i and p = 5. If i > 5, element at index i is (None, None).
        diff_lambda_global_new (list[(float, float)]): element at index i contains the mean and median errors (in that order) incurred when estimating Lambda^{(0)} using the general method and starting from the population cumulants, across 500 models with q = i and p = 5.
        diff_lambda_global_new_matrix (list[(float, float)]): element at index i contains the mean and median errors (in that order) incurred when estimating Lambda^{(0)} using the general method and starting from the matrices that should be obtained from tensor decomposition on the population cumulants, across 500 models with q = i and p = 5.
        diff_lambda_global_old (list[(float, float)]): if i <= 5, element at index i contains the mean and median errors (in that order) incurred when estimating Lambda^{(0)} using the injective method and starting from the population cumulants, across 500 models with q = i and p = 5. If i > 5, element at index i is (None, None).
        diff_lambda_global_old_matrix (list[(float, float)]): if i <= 5, element at index i contains the mean and median errors (in that order) incurred when estimating Lambda^{(0)} using the injective method and starting from the matrices that should be obtained from tensor decomposition on the population cumulants, across 500 models with q = i and p = 5. If i > 5, element at index i is (None, None).
        diff_graph_global_new (list[(float, float)]): element at index i contains the mean and median errors (in that order) incurred when estimating the latent graph using the general method and starting from the population cumulants, across 500 models with q = i and p = 5. Errors in the latent graph are calculated as given by helper_functions.graph_diff.
        diff_graph_global_new_matrix (list[(float, float)]): element at index i contains the mean and median errors (in that order) incurred when estimating the latent graph using the general method and starting from the matrices that should be obtained from tensor decomposition on the population cumulants, across 500 models with q = i and p = 5. Errors in the latent graph are calculated as given by helper_functions.graph_diff.
        diff_graph_global_old (list[(float, float)]): if i <= 5, element at index i contains the mean and median errors (in that order) incurred when estimating the latent graph using the injective method and starting from the population cumulants, across 500 models with q = i and p = 5. If i > 5, element at index i is (None, None). Errors in the latent graph are calculated as given by helper_functions.graph_diff.
        diff_graph_global_old_matrix (list[(float, float)]):  if i <= 5, element at index i contains the mean and median errors (in that order) incurred when estimating the latent graph using the injective method and starting from the matrices that should be obtained from tensor decomposition on the population cumulants, across 500 models with q = i and p = 5. If i > 5, element at index i is (None, None). Errors in the latent graph are calculated as given by helper_functions.graph_diff.
    """
    nlatent = np.linspace(2,7,6)
    plt.figure(figsize=(8, 6))
    plt.plot(nlatent, list(map(lambda x: x[0], diff_F_global_new)) , marker='o', linestyle='-', label='Tensor (general)')
    plt.plot(nlatent, list(map(lambda x: x[0], diff_F_global_new_matrix)) , marker='o', linestyle='-', label='Matrix (general)')
    plt.plot(nlatent, list(map(lambda x: x[0], diff_F_global_old)) , marker='o', linestyle='-', label='Tensor (injective)')
    plt.plot(nlatent, list(map(lambda x: x[0], diff_F_global_old_matrix)) , marker='o', linestyle='-', label='Matrix (injective)')
    plt.yscale("log")
    plt.xlabel('q')
    plt.ylabel('Median relative Frobenius error in F')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"figures/F.png")

    plt.figure(figsize=(8, 6))
    plt.plot(nlatent, list(map(lambda x: x[0], diff_lambda_global_new)) , marker='o', linestyle='-', label='Tensor (general)')
    plt.plot(nlatent, list(map(lambda x: x[0], diff_lambda_global_new_matrix)) , marker='o', linestyle='-', label='Matrix (general)')
    plt.plot(nlatent, list(map(lambda x: x[0], diff_lambda_global_old)) , marker='o', linestyle='-', label='Tensor (injective)')
    plt.plot(nlatent, list(map(lambda x: x[0], diff_lambda_global_old_matrix)) , marker='o', linestyle='-', label='Matrix (injective)')
    plt.yscale("log")
    plt.xlabel('q')
    plt.ylabel('Median relative Frobenius error in ' + r'$\Lambda^{(0)}$')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"figures/Lambda.png")

    plt.figure(figsize=(8, 6))
    plt.plot(nlatent, list(map(lambda x: x[0], diff_graph_global_new)) , marker='o', linestyle='-', label='Tensor (general)')
    plt.plot(nlatent, list(map(lambda x: x[0], diff_graph_global_new_matrix)) , marker='o', linestyle='-', label='Matrix (general)')
    plt.plot(nlatent, list(map(lambda x: x[0], diff_graph_global_old)) , marker='o', linestyle='-', label='Tensor (injective)')
    plt.plot(nlatent, list(map(lambda x: x[0], diff_graph_global_old_matrix)) , marker='o', linestyle='-', label='Matrix (injective)')
    plt.xlabel('q')
    plt.ylabel('Median error in graph')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"figures/graph.png")

def main():
    diff_F_global_new, diff_F_global_new_matrix, diff_F_global_old, diff_F_global_old_matrix, diff_lambda_global_new, diff_lambda_global_new_matrix, diff_lambda_global_old, diff_lambda_global_old_matrix, diff_graph_global_new, diff_graph_global_new_matrix, diff_graph_global_old, diff_graph_global_old_matrix = run()
    os.makedirs("figures",exist_ok=True)
    plot( diff_F_global_new, diff_F_global_new_matrix, diff_F_global_old, diff_F_global_old_matrix, diff_lambda_global_new, diff_lambda_global_new_matrix, diff_lambda_global_old, diff_lambda_global_old_matrix, diff_graph_global_new, diff_graph_global_new_matrix, diff_graph_global_old, diff_graph_global_old_matrix)

if __name__ == "__main__":
    main()