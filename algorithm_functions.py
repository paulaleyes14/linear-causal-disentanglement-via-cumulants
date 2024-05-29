# --- Imports ---
# Standard libraries
import sys
from math import factorial
from itertools import permutations, product

# Local
import helper_functions as hp

# Third-party
import causaldag as cd
import numpy as np
import tensorly as tl
from PyMoments import kstat
from SPM import subspace_power_method

residual_threshold = 0.00001
zero_threshold = 0.00001

# --- Functions to generate models and corresponding cumulants ---

def create_params(p, q):
    """
    Sample F and Lambdas.

    Args:
        p (int): number of observed variables.
        q (int): number of latent variables.

    Returns:
        F, Lambdas (tuple): a tuple containing the sampled F and Lambdas.
            F (numpy.ndarray): the mixing matrix, with shape (p, q).
            Lambdas (dict): a dictionary containing the observational and interventional Lambdas (perfect interventions).
                Keys indicate the context (str), and values are the corresponding Lambdas (numpy.ndarray).
    """
    Products = {}
    # Sample H
    rgen = np.random.default_rng()
    cond_number = 100
    while cond_number > 9:
        H = rgen.uniform(low=-2, high=2, size=(q,p))
        cond_number = np.linalg.cond(H)
    F = np.linalg.pinv(H)
    # Sample the graph and Lambda
    Lambdas = {}
    Lambda_obs = np.zeros((q,q))
    while np.linalg.norm(Lambda_obs) < 0.01:
        dag = cd.rand.directed_erdos(q, density=0.75, random_order=False)
        g = cd.rand.rand_weights(dag)
        Lambda_obs = g.to_amat()
    Lambdas["obs"]=Lambda_obs
    inv_obs = np.linalg.inv(np.eye(q)-Lambda_obs)
    product_obs = np.matmul(F,inv_obs)
    Products["obs"] = (product_obs, np.linalg.pinv(product_obs))
    omega_q1 = np.diag(np.ones(q))
    omega_q1[q-1,q-1]= rgen.uniform(low=2,high=3)
    product_q1 = np.matmul(np.matmul(F,inv_obs),omega_q1)
    Products["{0}".format(q-1)] = (product_q1, np.linalg.pinv(product_q1))
    Lambdas["{0}".format(q-1)] = Lambda_obs
    # Calculate interventional Lambdsas
    for i in range(q-1):
        new_Lambda = np.copy(Lambda_obs)
        new_Lambda[i] = 0
        Lambdas["{0}".format(i)] = new_Lambda
        inv = np.linalg.inv(np.eye(q)-new_Lambda)
        omega_new = np.diag(np.ones(q))
        omega_new[i,i] = rgen.uniform(low=2,high=3)
        product = np.matmul(np.matmul(F,inv),omega_new)
        Products["{0}".format(i)]= (product, np.linalg.pinv(product))

    return F, Lambdas, Products

def create_models(nmodels, nobserved, nlatent):
    """
    Generate random models. Each model is encoded as a tuple of its parameters (F, Lambdas).

    Args:
        nmodels (int): number of models to generate.
        nobserved (int): number of observed variables per model.
        nlatent (int): number of latent variables per model.
    
    Returns:
        model_params (list): a list containing the generated models. 
    """
    model_params = []
    for _ in range(nmodels):
        params = create_params(nobserved, nlatent)
        model_params.append(params)
    return model_params

def create_X_samples(nsamples, Lambda, F, i):
    """
    Generate sample data.

    Args:
        nsamples (int): number of samples of X to generate.
        Lambda (numpy.ndarray): matrix defining the latent graph.
        F (numpy.ndarray): mixing matrix.
        i (int): context indicator. Equals -1 in observational context and k in inteventional context corresponding to an intervention at Z_k.
        nonlinear_X (bool): if True, add nonlinearity in the transformation from latent to observed variables.
        alpha_X (float): coefficient quantifying amount of nonlinearity to add in the transformation from latent to observed variables. 
            Equals 0 if nonlinear_X is False.
        nonlinear_Z (bool): if True, add nonlinearity in the latent space.
        alpha_Z (float): coefficient quantifying amount of nonlinearity to add in the latent space. Equals 0 if nonlinear_Z is False.
    
    Returns:
        data (numpy.ndarray): the sample data.
        product (numpy.ndarray): the product F(I-Lambda)^{-1}D in the context specified by i.
    """
    p, q = np.shape(F)
    data = []
    rgen = np.random.default_rng()
    omega = np.diag(np.ones(q))
    I = np.eye(q)
    matrix = np.linalg.inv(I-Lambda)
    if i != -1:
        int_scale = rgen.uniform(low=1.25, high=4)
        omega[i,i] = factorial(2)*(int_scale**3)
    for _ in range(nsamples):
        epsilon_v = rgen.exponential(scale=1/np.cbrt(2),size=(q,1))
        if i != -1:
            epsilon_int = rgen.exponential(scale=int_scale)
            epsilon_v[i] = epsilon_int
        Z = np.matmul(matrix,epsilon_v)
        X = np.matmul(F,Z)
        X = np.reshape(X,(p,))
        data.append(X)
    data = np.asarray(data)
    omega_scaled = np.cbrt(omega)
    product = np.matmul(np.matmul(F,matrix),omega_scaled)

    return data, product

def sample_cumulant(nsamples, Lambda, F, i, order=3):
    """
    Calculate the sample third-order cumulant of X.
        Adapted from the code provided by Wang et al. in their paper "Identifiability of overcomplete independent component analysis".

    Args:
        nsamples (int): number of samples to use to calculate cumulant.
        Lambda (numpy.ndarray): matrix defining the latent graph.
        F (numpy.ndarray): mixing matrix.
        i (int): context indicator. Equals -1 in observational context and k in inteventional context corresponding to an intervention at Z_k.
        nonlinear_X (bool): if True, add nonlinearity in the transformation from latent to observed variables.
        alpha_X (float): coefficient quantifying amount of nonlinearity to add in the transformation from latent to observed variables. 
            Equals 0 if nonlinear_X is False.
        nonlinear_Z (bool): if True, add nonlinearity in the latent space.
        alpha_Z (float): coefficient quantifying amount of nonlinearity to add in the latent space. Equals 0 if nonlinear_Z is False.
    
    Returns:
        third_order_kstat (numpy.ndarray): the sample third-order cumulant.
        real_tensor (numpy.ndarray): the population third-order cumulant.
        diff (float): the Frobenius norm of the difference between the sample and the population cumulants.
        real_product (numpy.ndarray): the product F(I-Lambda)^{-1}D in the context specified by i.
    
    Citation:
        Kexin Wang and Anna Seigal. Identifiability of overcomplete independent component analysis. arXiv preprint arXiv:2401.14709, 2024.
    """
    # Calculate sample cumulant
    p, q = np.shape(F)
    data, real_product = create_X_samples(nsamples, Lambda, F, i)
    sym_indices, _, _ = hp.symmetric_indices(p,order)
    cumulants = np.apply_along_axis(lambda x: kstat(data,tuple(x)), 0, sym_indices)
    cumulants_dict = {tuple(sym_indices[:,n]): cumulants[n] for n in range(len(cumulants))}
    all_indices = np.array([list(i) for i in product(range(p), range(p),range(p))]) if order == 3 else np.array([list(i) for i in product(range(p), range(p),range(p), range(p))])
    values = np.apply_along_axis(lambda x: cumulants_dict[tuple(np.sort(x))], 1, all_indices)
    kstatfinal = values.reshape(p, p, p) if order == 3 else values.reshape(p,p,p,p)

    # Calculate population cumulant
    weights = np.ones(q)
    real_tensor = tl.cp_to_tensor((weights, [real_product, real_product, real_product])) if order == 3 else tl.cp_to_tensor((weights, [real_product, real_product, real_product, real_product]))
    diff = np.linalg.norm(kstatfinal-real_tensor)

    return kstatfinal, real_tensor, diff, real_product


def construct_cumulants_kstat(nsamples, Lambdas, F, order=3):
    """
    Construct cumulant tensors and products in all contexts.

    Args:
        nsamples (int): number of samples to use to calculate sample cumulants.
        Lambdas (dict): a dictionary containing the observational and interventional Lambdas (perfect interventions).
            Keys indicate the context (str), and values are the corresponding Lambdas (numpy.ndarray).
        F (numpy.ndarray): mixing matrix.
    
    Returns:
        Ts (dict): a dictionary containing the sample third-order cumulants.
            Keys indicate the context (str), and values are the corresponding sample cumulants (numpy.ndarray).
        realTs (dict): a dictionary containing the population third-order cumulants.
            Keys indicate the context (str), and values are the corresponding population cumulants (numpy.ndarray).
        Products (dict): a dictionary containing the products F(I-Lambda)^{-1}D across contexts.
            Keys indicate the context (str), and values are the corresponding products (numpy.ndarray).
        mean_diff (float): mean difference between population and sample cumulant across contexts.
    """
    Ts = {}
    realTs = {}
    Products = {}
    tensor_diffs = []
    keys = Lambdas.keys()
    for key in keys:
        i = -1 if key == "obs" else int(key)
        sample_tensor, real_tensor, diff, prod = sample_cumulant(nsamples, Lambdas[key], F, i, order)
        Ts[key] = sample_tensor
        realTs[key] = real_tensor
        Products[key] = prod
        tensor_diffs.append(diff)
    mean_diff = np.mean(tensor_diffs)
    return Ts, realTs, Products, mean_diff

# --- Functions for general case ---

def recoveringmatrix_cumulant(p,q,fourth_order_kstats):
    """
    Calculate the rank-q symmetric CP decomposition of an order-4 tensor using the subspace power method.

    Args:
        p (int): dimension of the tensor; the tensor to perform decomposition of is of size pxpxpxp.
        q (int): the rank of the decomposition.
        fourth_order_kstats (numpy.ndarray): tensor to decompose.
    """
    cols=subspace_power_method(fourth_order_kstats,n=4,d=p,r=q)
    def returnmindistancebewteenvectors(cols):
        def distancebewteenvectors(v1,v2):
            v1=v1.reshape(-1,1)
            v2=v2.reshape(-1,1)
            M=v1@np.transpose(v1)-v2@np.transpose(v2)
            return np.sum(M*M)
        lens=cols.shape[1]
        error=1
        for i in range(lens):
            for j in range(i+1,lens):
                error=min(error,distancebewteenvectors(cols[:,i],cols[:,j]))
        return error
    step=0
    while returnmindistancebewteenvectors(cols)<0.1 and step<1000:
        cols=subspace_power_method(fourth_order_kstats,n=4,d=p,r=q)
        step+=1
    return cols

def find_permutation(obs_matrix, int_matrix, even=False):
    """
    Given two matrices obs_matrix and int_matrix, find permutation (and signs if even is True) of the columns of int_matrix minimizing the rank of obs_matrix-int_matrix.

    Args:
        obs_matrix (numpy.ndarray): matrix that remains unchanged.
        int_matrix (numpy.ndarray): matrix whose columns should be permuted (and their sign modified)

    Returns:
        The product of int_matrix and P, where P is the permutation of the columns of int_matrix minimizing the rank of obs_matrix-int_matrix.
    """
    _, q = np.shape(obs_matrix)
    sing_value = sys.float_info.max
    global_match = None
    global_scaling = None
    if even:
        scaling_list = list(product([1,-1], repeat=q))
    column_matrix = int_matrix.T.tolist()
    columns = list(range(len(column_matrix)))
    for perm in permutations(columns):
        shuffled_column_matrix = []
        for idx in perm:
            shuffled_column_matrix.append(column_matrix[idx])
        new_matrix = np.array(shuffled_column_matrix).T
        # If even is true, we consider different signs of columns
        if even:
            permuted_matrix = new_matrix.copy()
            for scale in scaling_list:
                new_matrix = np.matmul(permuted_matrix, np.diag(scale))
                sub_matrix = obs_matrix - new_matrix
                _, S, _ = np.linalg.svd(sub_matrix)
                Sabs = [abs(x) for x in S]
                Ssorted = sorted(Sabs, reverse=True)
                new_sing_value = Ssorted[1]
                if new_sing_value < sing_value:
                    sing_value = new_sing_value
                    global_match = perm
                    global_scaling = np.diag(scale)
        else:
            sub_matrix = obs_matrix - new_matrix
            _, S, _ = np.linalg.svd(sub_matrix)
            Sabs = [abs(x) for x in S]
            Ssorted = sorted(Sabs, reverse=True)
            new_sing_value = Ssorted[1]
            if new_sing_value < sing_value:
                sing_value = new_sing_value
                global_match = perm
    P = np.zeros((q,q))
    for i in range(q):
        orig_column = global_match[i]
        P[orig_column][i] = 1
    if even:
        return np.matmul(np.matmul(int_matrix, P), global_scaling)
    else:
        return np.matmul(int_matrix, P)

def recover_int_target(matrix_obs, matrix_int, current_ints, threshold):
    """
    Determine the intervention target of context where matrix_int was recovered.

    Args:
        matrix_obs (numpy.ndarray): product recovered from tensor decomposition in the observational context.
        matrix_int (numpy.ndarray): product recovered from tensor decomposition in an interventional context.
        current_ints (set[int]): a set containing the intervention targets that have already been matched with an interventional context.
        threshold (float): numbers under this value will be considered zero.
    
    Returns:
        Letting k denote the context where matrix_int was recovered, the function returns a tuple containing the cube root of the third-order 
        cumulant of epsilon_ik in the kth context, the norm of the vector obtained when subtracting the ik^th column of matrix_obs from the projection of the
        ik^th column of matrix_int onto the ik^th column of matrix_obs, and ik. 
    """
    _, q = np.shape(matrix_obs)
    scale_res = []
    for i in range(q):
        col_obs = matrix_obs[:,i].copy()
        col_int = matrix_int[:,i].copy()
        proj = col_obs * np.dot(col_obs, col_int)/np.dot(col_obs, col_obs)
        res = np.linalg.norm(proj - col_int)
        scale = proj[0]/col_obs[0]
        scale_res.append((scale, res, i))
    filtered_list = filter(lambda x: abs(x[1]) <= threshold, scale_res)
    sorted_list = sorted(filtered_list, reverse=True, key = lambda x: abs(1 - abs(x[0])))
    for tuple in sorted_list:
        if tuple[2] not in current_ints:
            return tuple
    filtered_list_int = filter(lambda x: x[2] not in current_ints, scale_res)
    return sorted(filtered_list_int, key = lambda x: abs(x[1]))[0]

def align_outputs(products, even=False):
    """
    Given a dictionary containing the products recovered from tensor decomposition in the different contexts, match each with its intervention target, standardize labeling
    of latent nodes across contexts, and undo scaling resulting from the change in the stochasticity of epsilon_ik in the kth interventional context.

    Args:
        products (dict): a dictionary containing the products recovered in each context.
            Keys indicate the context (str), and values are the corresponding products (numpy.ndarray).
        even (bool): boolean indicating whether the products were recovered from tensor decomposition of an even (even=True) or odd (even=False) cumulant. Default is false.

    Returns:
        new_tuples (dict): a dictionary containing the products recovered from tensor decomposition in the different contexts, after standardizing the labeling of latent nodes 
        across contexts and undoing the scaling resulting from the change of stochasticity of epsilon_ik in the k^th interventional context, together with their corresponding intervention targets.
            Keys indicate the context (str), and values are tuples containing the corresponding recovered product after standardizing the labeling of latent nodes, the product
            after both standardizing the labeling of latent nodes and undoing the scaling, and the corresponding intervention target.
    """
    keys = products.keys()
    matrix_obs = products["obs"]
    current_ints = set()
    new_tuples = {}
    new_tuples["obs"] = (matrix_obs, -1)
    for key in keys:
        if key == "obs":
            continue
        else:
            new_product = find_permutation(matrix_obs, products[key],even)
            s, _, int = recover_int_target(matrix_obs, new_product, current_ints, residual_threshold)
            current_ints.add(int)
            new_product_noscale = new_product.copy()
            new_product[:,int] = new_product[:,int]*1/s
            new_tuples[key] = (new_product_noscale, new_product, int)
    return new_tuples


def recover_lambda(tuples, zero_threshold):
    """
    Recover lambda, the matrix encoding the latent graph.

    Args:
        tuples (dict): a dictionary containing the products recovered from tensor decomposition in the different contexts, after standardizing the labeling of latent nodes 
        across contexts and undoing the scaling resulting from the change of stochasticity of epsilon_ik in the k^th interventional context, together with their corresponding intervention targets.
            Keys indicate the context (str), and values are tuples containing the corresponding recovered product after standardizing the labeling of latent nodes, the product
            after both standardizing the labeling of latent nodes and undoing the scaling, and the corresponding intervention target.
        zero_threshold (float): numbers whose absolute value is below this value are considered to be zero.
    """
    matrix_obs = tuples["obs"][0]
    _, q = np.shape(matrix_obs)
    iminuslinv = np.zeros((q,q))
    keys = tuples.keys()
    for key in keys:
        if key == "obs":
            continue
        else:
            _, product, k = tuples[key]
            for j in range(q):
                if k == j:
                    iminuslinv[k,j] = 1
                else:
                    iminuslinv[k,j]=(matrix_obs[0,j]-product[0,j])/matrix_obs[0,k]
    iminuslinv[abs(iminuslinv) < zero_threshold] = 0
    iminusl = np.linalg.inv(iminuslinv)
    lambdam = (-1)*(iminusl-np.eye(q))
    lambdam[abs(lambdam) < zero_threshold] = 0
    return iminuslinv, lambdam

def recover_F(matrix_obs, Lambda):
    """
    Recover F, the mixing matrix.

    Args:
        matrix_obs (numpy.ndarray): product recovered from tensor decomposition in the observational context.
        Lambda (numpy.ndarray): matrix encoding the latent graph.
    """
    _, q = np.shape(matrix_obs)
    mat = np.eye(q)-Lambda
    return np.matmul(matrix_obs,mat)

# --- Functions for injective case ---

def simult_diag(T, q):
    """
    Calculate the rank-q symmetric CP decomposition of an order-3 tensor using simultaneous diagonalization.

    Args:
        T (numpy.ndarray): the tensor to perform tensor decomposition on.
        q (int): the rank of the decomposition.
    
    Returns:
        numpy.ndarray: the normalized (by column) factor matrix of the decomposition.
    """
    p, _, _ = np.shape(T)
    a = np.random.rand(p)
    anorm = a/np.linalg.norm(a)
    b = np.random.rand(p)
    borth = b - np.inner(b,a)*a
    bnorm = borth/np.linalg.norm(borth)

    scaled_tensor_a = np.einsum('ijk,i->ijk', T, anorm)
    Ma = np.sum(scaled_tensor_a,axis=0)
    scaled_tensor_b = np.einsum('ijk,i->ijk', T, bnorm)
    Mb = np.sum(scaled_tensor_b,axis=0)

    prod = np.matmul(Ma, np.linalg.pinv(Mb))
    eigvalues, eigvectors = np.linalg.eig(prod)
    idx = np.argsort(np.abs(eigvalues))
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:,idx]

    return np.real_if_close(eigvectors[:,-q:], tol=1)


def find_alphas(T, recovered_product):
    """
    Calculate tensor decomposition coefficients (required when tensor decomposition is performed using simultaneous diagonalization).

    Args:
        T (numpy.ndarray): the tensor whose decomposition's coefficients we want to calculate.
        recovered_product (numpy.ndarray): the normalized (by column) factor matrix recovered by performing tensor decomposition on T.
    
    Returns:
        alphas (numpy.ndarray): the coefficients minimizing the Frobenius norm of the difference between T and the reconstructed tensor.
    """
    p, q = np.shape(recovered_product)
    b = np.reshape(T, p**3)
    A = []
    for j in range(q):
        current_v = np.reshape(recovered_product[:,j],(p,1))
        current_tensor = tl.cp_to_tensor((np.ones(1),[current_v,current_v,current_v]))
        new_column = np.reshape(current_tensor,p**3)
        A.append(new_column)
    Amat = np.transpose(np.array(A))
    alphas, _, _, _  = np.linalg.lstsq(Amat, b, rcond=None)
    return alphas

def nonnorm_products(Ts, RProducts):
    """
    Given a dictionary of tensors and the normalized factor matrices of their rank-r symmetric CP decompositions, 
        calculate the non-normalized factor matrices and their pseudoinverses.

    Args:
        Ts (dict): a dictionary containing the cumulants in each context.
            Keys indicate the context (str), and values are the corresponding cumulants (numpy.ndarray).
        RProducts (dict): a dictionary containing the normalized (by column) factor matrices recovered by decomposing the tensors in each context.
            Keys indicate the context (str), and values are the corresponding normalized factor matrices (numpy.ndarray).
        
    Returns:
        RProducts_nn (dict): a dictionary containing the non-normalized factor matrices and their pseudoinverses.
            Keys indicate the context (str), and values are tuples containing the non-normalized factor matrices (numpy.ndarray) and their pseudoinverses (numpy.ndarray).
    """
    _, q = np.shape(RProducts["obs"])
    RProducts_nn = {}
    for key in Ts.keys():
        current_product = RProducts[key].copy()
        current_alphas = find_alphas(Ts[key],current_product)
        for i in range(q):
            current_product[:,i] = hp.cube(current_alphas[i])*current_product[:,i]
        current_product_inv = np.linalg.pinv(current_product)
        RProducts_nn[key] = (current_product, current_product_inv)
    return RProducts_nn

def recover_int_target_injective(C, Ctilde, current_ints):
    """
    Determine intervention target and permutation matrix of interventional context where pseudoinverse of Ctilde was recovered.

    Args:
        C (numpy.ndarray): pseudoinverse of the product recovered in the observational context.
        Ctilde (numpy.ndarray): pseudoinvere of the product recovered in an interventional context with unknown intervention target.
        current_ints (set[int]): a set containing the intervention targets of the interventional contexts that have already been matched with an intervention target.
    
    Returns:
        tuple (tuple): a tuple containing three elements: (i, j, diff).
            i (int): intervened variable in the context where the pseudoinverse of Ctilde was recovered.
            j (int): index indicating the relabeling of Zi in the context where the pseudoinverse of Ctilde was recovered.
            diff (float): Euclidean norm of the difference between C[i,:] and Ctilde[j,:]
        P (numpy.ndarray): the permutation matrix encoding the relabeling of the latent nodes in the interventional context where the pseudoinverse of Ctilde was recovered.
    """
    q, _ = np.shape(C)
    diff_list = []
    matches_int = {}
    P = np.zeros((q,q))
    for k in range(q):
        matches_int[k] = 0

    # Initial matching of rows of C with rows of \tilde{C}
    for i in range(q):
        row_int, diff = hp.match_row(C[i,:].copy(), Ctilde)
        matches_int[row_int] += 1
        diff_list.append((i, row_int, diff))
    
    # Rematch incorrectly matched rows
    unmatched_rows = [k for k, v in matches_int.items() if v == 0]
    for (key, value) in matches_int.copy().items():
        if value > 1:
            tuple_list = list(filter(lambda x: x[1]==key,diff_list))
            tuple_list.sort(key = lambda x: x[2])
            tuple_list.remove(tuple_list[0])

            diff_list = [elt for elt in diff_list if elt not in tuple_list]
            
            for obs_row_ind, _, _ in tuple_list:
                row_int_new, diff_new = hp.match_row(C[obs_row_ind,:].copy(), Ctilde[unmatched_rows,:])
                new_match = (obs_row_ind, unmatched_rows[row_int_new], diff_new)
                unmatched_rows.remove(unmatched_rows[row_int_new])
                diff_list.append(new_match)
    
    diff_list.sort(reverse=True, key=lambda x: x[2])
    for (i, j, _) in diff_list:
        P[i,j]=1
    for tuple in diff_list:
        if not(tuple[0] in current_ints):
            return tuple, P

def match_int_injective(Productsnn):
    """
    Determine the intervention target and undo relabeling of latent nodes arising from tensor decomposition in each interventional context.

    Args:
        Productsnn (dict): a dictionary containing the non-normalized factor matrices recovered via tensor decomposition and least squares in each context.
            Keys indicate the context (str), and values are tuples containing the corresponding non-normalized factor matrix (numpy.ndarray) and its pseudoinverse (numpy.ndarray).

    Returns:
        tuples (dict): a dictionary mapping each context to its non-normalized factor matrix, its pseudoinverse (both after undoing relabeling of latent nodes) and its intervention target. 
            Keys indicate the context (str), and values are tuples containing the corresponding non-normalized factor matrix (numpy.ndarray), its pseudoinverse (numpy.ndarray),
                and a tuple containing the intervention target of the context (int) and the relabeling of the intervened variable in the context (int). The latter two are both -1 in the observational context.
    """
    tuples = {}
    Prodnn, C = Productsnn["obs"]
    tuples["obs"] = (Prodnn, C, (-1,-1))
    keys = Productsnn.keys()
    current_ints = set()
    for key in keys:
        if key == "obs":
            continue
        else:
            Prodtildenn, Ctilde = Productsnn[key]
            _, q = np.shape(Prodtildenn)
            tuple, P = recover_int_target_injective(C, Ctilde, current_ints)

            assert(np.linalg.matrix_rank(P)==q)
            current_ints.add(tuple[0])
            Prodtildenn_new = np.matmul(Prodtildenn, np.transpose(P))
            Cprimetilde = np.matmul(P, Ctilde)
            tuples[key] = (Prodtildenn_new, Cprimetilde, (tuple[0], tuple[1]))
    return tuples

def recover_Dprime(tuples):
    """
    Recover the third-order moment of the noise variables across contexts.

    Args:
        tuples (dict): a dictionary containing the factor matrices recovered via tensor decomposition in each context (with consistent labeling of latent nodes across contexts),
            its pseudoinverses and the intervention targets.
                Keys indicate the context (str), and values are tuples containing the corresponding factor matrix (numpy.ndarray), its pseudoinverse (numpy.ndarray), and the intervention target ik of the context (int) 
                (set to -1 in the observational context).
    
    Returns:
        new_tuples (dict): a dictionary containing the factor matrices recovered via tensor decomposition in each context (with consistent labeling of latent nodes across contexts),
            its pseudoinverses, the intervention targets, and the corresponding interventional scalings.
                Keys indicate the context (str), and values are tuples containing the corresponding factor matrix (numpy.ndarray), its pseudoinverse (numpy.ndarray), the intervention target ik of the context (int) 
                (set to -1 in the observational context), and the cube root of epsilon_ik in the interventional context with intervention target ik (int).
    """
    keys = tuples.keys()
    new_tuples = {}
    for key in keys:
        if key == "obs":
            Prodnn, C, nonint = tuples["obs"]
            new_tuple_obs = (Prodnn, C, nonint, 1)
            new_tuples[key] = new_tuple_obs
        else:
            Prodnntilde, Ctildenn, int_target = tuples[key]
            a, _ = int_target
            col_obs = Prodnn[:,a].copy()
            col_int = Prodnntilde[:,a].copy()
            proj = col_obs * np.dot(col_obs, col_int)/np.dot(col_obs, col_obs)
            Diiprime = proj[0]/col_obs[0]
            new_tuple = (Prodnntilde, Ctildenn, int_target, Diiprime)
            new_tuples[key] = new_tuple
    return new_tuples

def recover_H(tuples):
    """
    Recover the mixing matrix.

    Args:
        tuples (dict): a dictionary containing the factor matrices recovered via tensor decomposition in each context (with consistent labeling of latent nodes across contexts),
            its pseudoinverses, the intervention targets, and the corresponding interventional scalings.
                Keys indicate the context (str), and values are tuples containing the corresponding factor matrix (numpy.ndarray), its pseudoinverse (numpy.ndarray), the intervention target ik of the context (int) 
                (set to -1 in the observational context), and the cube root of epsilon_ik in the interventional context with intervention target ik (int).
    
    Returns:
        H (numpy.ndarray): the pseudoinverse of the mixing matrix F.
    """
    keys = tuples.keys()
    Hpair = []
    for key in keys:
        if key == "obs":
            continue
        else:
            _, Cprimea, int_target, Diiprime = tuples[key]
            a, _ = int_target
            new_row = Diiprime * Cprimea[a,:].copy()
            row_pair = (new_row, a)
            Hpair.append(row_pair)
    Hpair.sort(key=lambda x: x[1])
    H = list(map(lambda x: x[0], Hpair))
    return np.array(H)

def recover_lambda_injective(matrix_obs, H, zero_threshold):
    """
    Recover lambda, the matrix encoding the latent graph.

    Args:
        matrix_obs (numpy.ndarray): product recovered from tensor decomposition in the observational context.
        H (numpy.ndarray): pseudoinverse of the mixing matrix.
        zero_threshold (float): numbers whose absolute value is below this value are considered to be zero.
    
    Returns:
        lambdar (numpy.ndarray): the matrix encoding the latent graph.
    """
    _, q = np.shape(matrix_obs)
    iminuslinv = np.matmul(H,matrix_obs)
    iminusl = np.linalg.inv(iminuslinv)
    lambdar = (-1)*(iminusl - np.eye(q))
    lambdar[abs(lambdar) < zero_threshold] = 0
    return lambdar

# --- Common recovery functions ---

def decompose_tensors(Ts,p,q,order=3):
    """
    Calculate the rank-q symmetric CP decomposition of the cumulants in all contexts, 
    using either simultaneous diagonalization or the subspace power method.

    Args:
        Ts (dict): a dictionary containing the cumulants to decompose.
            Keys indicate the context (str), and values are the corresponding cumulants (numpy.ndarray).
        p (int): dimensionality of the tensors.
        q (int): rank of the decomposition.
        order (int): order of the tensors to decompose. Default is 3.
    
    Returns:
        Product_recovery (dict): a dictionary containing the recovered factor matrices.
            Keys indicate the context (str), and values are the corresponding factor matrices (numpy.ndarray).
    """
    Product_recovery = {}
    for key in Ts.keys():
        factor = simult_diag(Ts[key], q) if order == 3 else recoveringmatrix_cumulant(p,q,Ts[key])
        Product_recovery[key] = factor
    return Product_recovery

def calculate_product_diff(tuples, Products, perm):
    keys = Products.keys()
    diff_list = []
    for key in keys:
        if key == "obs":
            continue
        else:
            Prodtilde, _, _ = tuples[key]
            Prod_after = np.matmul(Prodtilde, perm)
            diff = np.linalg.norm(Products[key]-Prod_after)
            diff_list.append(diff)
    return np.mean(diff_list)