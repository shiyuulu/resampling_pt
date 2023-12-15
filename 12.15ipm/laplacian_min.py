import numpy as np 
import scipy.sparse
from collections import OrderedDict
import open3d as o3d

def sample_points(points, scores, n_samples, method):
    if method == 'prob':
        return sample_points_prob(points, scores, n_samples)
    elif method == 'topk':
        return sample_points_topk(points, scores, n_samples)

def sample_points_topk(pcd, scores, n_samples):
    """ sample points with top K scores.

    Args:
        points: np.ndarray 
            shape (N, 3)

        scores: np.ndarray
            score for each point . shape (N,)

        n_samples: int
            number of sampled points

    Returns:
        points_sampled: np.ndarray
            shape (n_samples, 3)
        
    """
   
    # Find indices where scores are not 1
    points = np.asarray(pcd.points)
    scores=np.asarray(scores)
    valid_indices = np.where(scores != 1)[0]

    # Sort valid indices by scores in descending order and select top n_samples
    top_k_indices = np.argsort(-scores[valid_indices])[:n_samples]

    # Select the corresponding points
    points_sampled = pcd.select_by_index(valid_indices[top_k_indices])
    # points[valid_indices[top_k_indices]]


    return points_sampled

def sample_points_prob(points, scores, n_samples):
    """ sample points according to score normlized probablily

    Args:
        points: np.ndarray 
            shape (N, 3)

        scores: np.ndarray
            score for each point . shape (N,)

        n_samples: int
            number of sampled points

    Returns:
        points_sampled: np.ndarray
            shape (n_samples, 3)
        
    """
    scores=np.asarray(scores)
    valid_indices = np.where(scores != 1)[0]
    scores=scores[valid_indices]
    scores = scores / np.sum(scores)
    ids_sampled = np.random.choice(
        len(valid_indices), n_samples, replace=False, p=scores)
    points_sampled = points[ids_sampled]
    return points_sampled

def compute_resolution(kd_tree, pcd_np, scale_min_dist, scale_max_dist):
    """ compute radius for search
    Args:
        kdtree: o3d.geometry.KDTreeFlann
            kdtree for point cloud

        pcd_np: np.ndarray
            shape (N, 3)

        scale_min_dist: int/float
            scale the min distance to be searching radius

        scale_max_dist: int/float
            scale the max distance to be searching radius

    Returns:
        radius: float
            radius for searching neighbors
    """

    min_dist = np.inf
    max_dist = -np.inf
    N = pcd_np.shape[0]
    for i in range(N):
        [k, idx, dist_square] = kd_tree.search_knn_vector_3d(pcd_np[i], 2) # dist is squared distance
        min_dist = min(min_dist, dist_square[1])
        max_dist = max(max_dist, dist_square[1])
    min_dist = np.sqrt(min_dist)
    max_dist = np.sqrt(max_dist)
    
    radius = min(min_dist * scale_min_dist, max_dist * scale_max_dist)
    print(f"min_dist: {min_dist}, max_dist: {max_dist}, radius: {radius}")

    return radius
from tqdm import tqdm



def compute_local_curvature(cov_matrix):
    """
    Compute local curvature using PCA on the neighborhood.
    """
    # mean = np.mean(neighbors, axis=0)
    #cov_matrix = np.cov(neighbors, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)
    smallest_eigenvalue = eigenvalues[sorted_indices[0]]

    return smallest_eigenvalue / (eigenvalues.sum())

def fit_plane_least_squares(points):
    """
    Fit a plane using least squares method for a given set of points.
    Returns the plane coefficients (a, b, c, d) for the plane equation ax + by + cz + d = 0.
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[2, :]
    d = -np.dot(normal, centroid)
    return np.append(normal, d)



def compute_adjacency_matrix(kd_tree, pcd_np,pcd_cov, radius, max_neighbor, var=None):
    """ compute adjacency matrix 
    Notice: for very sparse point cloud, you may not find any neighbor within radius,
    Then the row for this point in matrix is all zero. Will cause bug. 
    Maybe replace search_hybrid_vector_3d with search_knn_vector_3d
    

    Args:
        kdtree: o3d.geometry.KDTreeFlann(pcd)
            kdtree for point cloud

        pcd_np: np.ndarray
            shape (N, 3)

        radius: float
            searching radius for nearest neighbor

        max_neighbor: int
            max number of nearest neighbor

        var: float


    Returns:
        adj_matrix_new: scipy.sparse.coo_matrix

    """
    N = pcd_np.shape[0]
    #adj_dict = OrderedDict()
    adj_matrix = scipy.sparse.dok_matrix((N, N))
    
   
    # points = np.asarray(pcd_np.points)
   # distances = np.zeros(N)
    dis=[]
    
    for i in tqdm(range(N)):
        [k, idx, dist_square] = kd_tree.search_hybrid_vector_3d(pcd_np[i], radius, max_neighbor)
        
        # curv=compute_local_curvature(pcd_cov[i])
    
        dist_square_value = np.asarray(dist_square)[0]
        # * np.exp(-(curv*curv)/0.0004)

        adj_matrix[i, idx] = np.exp(-dist_square_value / radius**2) 

     
    adj_matrix = adj_matrix.tocoo()
    row, col, dist_square = adj_matrix.row, adj_matrix.col, adj_matrix.data

    data_new = np.exp(- dist_square )
    adj_matrix_new = scipy.sparse.coo_matrix((data_new, (row, col)), shape=(N,N))

    return adj_matrix_new, dis


from scipy.sparse import csr_matrix
def compute_D(W):
    """ compute degree matrix
    Args:
        W: scipy.sparse.coo_matrix

    Returns:
        D: sparse.matrix
    """
    N = W.shape[0]

    diag = np.array(W.sum(axis=1)).flatten()
    D = scipy.sparse.coo_matrix((N, N))
    D.setdiag(diag)

    return D

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize

def laplacian_smoothing(pcd_np, L,A, features, alpha=0.001, num_iter=100):
    """
    Apply Laplacian smoothing to a point cloud.

    Args:
    pcd_np (np.ndarray): The original point cloud, shape (N, 3).
    L (np.ndarray): The graph Laplacian, shape (N, N).
    alpha (float): Regularization parameter.
    num_iter (int): Number of iterations for the optimization.

    Returns:
    np.ndarray: The smoothed point cloud, shape (N, 3).
    """
    N = pcd_np.shape[0]

    gamma=0.05
    num_iter=800
    # Objective function: Minimize x^T L x + alpha * ||x - x0||^2
    S = diags(features)

      # Construct the Psi vector for uniformity loss
    Psi = np.ones(N)  # Assuming the desired density is uniform

    # Create a mask for non-feature points
    non_feature_mask = np.ones(N, dtype=bool)
    non_feature_mask[features] = False
    feature_mask = np.ones(N, dtype=bool)
    feature_mask[features] = True

    k=30


    def objective(x):
        x = x.reshape(N, 3)
        laplacian_term =  gamma * np.sum((L.dot(x))**2)  
        
        # np.sum(L.dot(x)* x  )     #laplacan loss
        #np.sum((x - pcd_np)**2)  #distance loss
        # np.sum((x - L.dot(x))**2)  #feature loss
        #np.sum(x.T@L@ x)            # #total vari
        # np.sum((L.dot(x))**2)       #smooth
        regularization_term = (1-gamma)*(np.linalg.norm(S @ (x - pcd_np))**2 +np.sum((x - pcd_np)**2))
        # +np.sum((x - pcd_np)**2)

        # uniform_term= (1-gamma)*np.sum((A.dot(Psi) - alpha * k)**2)

        return laplacian_term + regularization_term 

    def gradient(x):
        x = x.reshape(N, 3)
        laplacian_grad =  gamma * 2 * L.T.dot(L.dot(x)) 
        # 2 * L.dot(x)              #laplacan loss
        # (x - pcd_np)                 #distance loss
        # L.T.dot(L.dot(x) - x)        #feature loss
        # L.dot(x)                     #total vari
        # *L.T.dot(L.dot(x))            #smooth
        regularization_grad =(1-gamma) *2 *(S @ (x - pcd_np)+(x - pcd_np) )
        #uniformity_grad = 2 * (1-gamma) * (A.T.dot(A.dot(Psi) - alpha * k))[:, np.newaxis]
        # uniformity_term = 2 * (1 - gamma) * (A.T.dot(A.dot(Psi) - alpha * k))[:, np.newaxis]

        return (laplacian_grad + regularization_grad).flatten()
    # + uniformity_term



    # def objective(x):
    #     x = x.reshape(N, 3)
    #     laplacian_term =  gamma * np.sum((L.dot(x))**2)  
        
    #     # np.sum(L.dot(x)* x  )     #laplacan loss
    #     #np.sum((x - pcd_np)**2)  #distance loss
    #     # np.sum((x - L.dot(x))**2)  #feature loss
    #     #np.sum(x.T@L@ x)            # #total vari
    #     # np.sum((L.dot(x))**2)       #smooth
    #     regularization_term = (1-gamma)*np.sum((x - pcd_np)**2)

    #     return laplacian_term + regularization_term

    # def gradient(x):
    #     x = x.reshape(N, 3)
    #     laplacian_grad =  gamma * 2 * L.T.dot(L.dot(x))   
    #     # 2 * L.dot(x)              #laplacan loss
    #     # (x - pcd_np)                 #distance loss
    #     # L.T.dot(L.dot(x) - x)     #feature loss
    #     # L.dot(x)                     #total vari
    #     # *L.T.dot(L.dot(x))            #smooth
    #     regularization_grad =(1-gamma) *2 *(x - pcd_np)
    #     return (laplacian_grad + regularization_grad).flatten()

    # @works well:
    # smooth/ total ari +distance loss regularization.  gamma= 0.1

    # Initial guess
    x0 = pcd_np.flatten()

    # Optimize
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': num_iter})
    
    
    if res.success:
        message = "Optimization successful."
    else:
        message = "Optimization reached the maximum number of iterations."

    # pt=np.array(res.x.reshape(N, 3))
    # s_pt=np.array(res.x.reshape(N, 3))[features==0]
    # f_pt=pcd_np[features==1]

    # pt=np.vstack([s_pt,f_pt])


    return res.x.reshape(N, 3)
   



def apply_filter(pcd_np, F, FN):
    """ 
    Args:
        pcd_np: np.ndarray 
            shape (N, 3)

        F: sparse.coo_matrix (N, N)
            graph filter

    Returns:
        scores: np.ndarray
            shape (N,)
    """
    scores_pt =FN @  pcd_np#  (N, 3).  X_i - sum( weight_ij * neighbor(X_i)_j )

    scores=F@  pcd_np
    scores = np.linalg.norm(scores, ord=2, axis=1)  # [N], L2 distance
    scores = scores ** 2  # [N], L2 distance square 

    scores=np.asarray(scores)  
    # valid_indices = np.where(scores != 1)[0
    n_samples=int(0.6*len(pcd_np))

    n_sam=int(0.3*len(pcd_np))
    # Sort valid indices by scores in descending order and select top n_samples
    sampled = np.argsort(scores)[:n_samples]

    sam=np.argsort(-scores)[:n_sam]

    # filtered_scores_pt = np.vstack([scores_pt[sampled],pcd_np[sam]]) 
    filtered_scores_pt = pcd_np[sam]
    # filtered_scores = scores[mask]

    # Initialize the binary array with zeros
    features = np.zeros(len(pcd_np), dtype=int)  # N should be the total number of points in your point cloud

    # Set the entries at the feature indices to 1
    features[sam] = 1


    print(len(scores_pt))
    print(len(scores))
    print(len(filtered_scores_pt))

    return filtered_scores_pt, scores,features

from scipy.sparse.linalg import inv

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

def compute_graph_laplacian(W):
    D = diags(W.sum(axis=1).A1, 0)
    L = D - W
    return L
from scipy.linalg import eigh





def plot_signal_distribution(transformed_signal, eigenvalues):
    # Compute magnitude of the transformed signal
    magnitudes = np.abs(transformed_signal)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.stem(eigenvalues, magnitudes, use_line_collection=True)
    plt.xlabel('Frequency (Eigenvalues of Laplacian)')
    plt.ylabel('Magnitude of Transformed Signal')
    plt.title('Signal Distribution in the Spectral Domain')
    plt.show()


def compute_scores_from_points(pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var):
    """ compute scores for each point.

    """
  

    pcd_np = np.asarray(pcd.points)
    N = pcd_np.shape[0]
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    radius = 0.3

    pcd.estimate_covariances( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbor))
    pcd_cov=pcd.covariances

    pcd.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbor))
    pcd_nm=np.asarray(pcd.normals)

    # compute_resolution(kd_tree, pcd_np, scale_min_dist, scale_max_dist)

    W,dis = compute_adjacency_matrix(kd_tree, pcd_np,pcd_cov, radius, max_neighbor, var)
    D = compute_D(W) 

    
    if filter_type == "all":
        F = scipy.sparse.coo_matrix((N, N))
        F.setdiag(1)

    elif filter_type == "high":
      
        F=D-W
     
        # Compute the degree matrix D
        row, col, data = D.row, D.col, D.data
        data = np.array(data)
        data[data == 0] = 1
        data_inv = 1/data
        D_inv =  scipy.sparse.coo_matrix((data_inv, (row, col)), shape=(N, N))
        # Apply the filter: F = D^-1 * W
        FN = D_inv @ W

    # scores_pt,scores = apply_filter(pcd_np, F,FN)
    sample_pt,scores, feature = apply_filter(pcd_np, F,FN)

    # scores_pt =  opti(pcd_np, F)
    pcd_fea = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample_pt))

    pcd_fea.paint_uniform_color([0,0,1])

    o3d.visualization.draw_geometries([pcd_fea])

    scores_pt=laplacian_smoothing(pcd_np, F,W, feature)

    pcd_af = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scores_pt))

    pcd.paint_uniform_color([1,0,0])

    pcd_af.paint_uniform_color([0,1,0])

    o3d.visualization.draw_geometries([pcd_af])

    o3d.visualization.draw_geometries([pcd_af,pcd,pcd_fea])
    # scores=transformed_signal

    return scores_pt, scores, dis


import matplotlib.pyplot as plt

def create_filter_and_stretch_histogram(data, threshold):

 # Create histogram bins
    hist, bin_edges = np.histogram(data, bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot original histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=256, range=(0, 1), color='blue', alpha=0.7)
    plt.title("Original Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    # Determine which bins to keep
    bins_to_keep = hist >= threshold

    # Update data: Set values from filtered-out bins to 1, and keep others as is
    updated_data = np.copy(data)
    for keep, bin_start, bin_end in zip(bins_to_keep, bin_edges, bin_edges[1:]):
        if not keep:
            updated_data[(data >= bin_start) & (data < bin_end)] = 1

    # Filter data for stretching
    data_for_stretching = updated_data[updated_data != 1]

    # Stretching the histogram
    if data_for_stretching.size > 0:
        min_value = np.min(data_for_stretching)
        max_value = np.max(data_for_stretching)
        scale = 1/ (max_value - min_value)
        updated_data[updated_data != 1] = (data_for_stretching - min_value) * scale


    # Plot stretched histogram
    plt.subplot(1, 2, 2)
    plt.hist(updated_data, bins=256, range=(0, 1), color='red', alpha=0.7)
    plt.title("Stretched Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


    return updated_data


 
def paint_points(ptCloud, voxel_distances):
    points = np.asarray(ptCloud.points)
    colors = np.zeros_like(points)

    # Flatten all distances to find global min and max
    all_distances =list(voxel_distances)
    min_dist, max_dist = np.min(all_distances), np.max(all_distances)


    print(max_dist)
    print(min_dist)
    print(np.mean(all_distances))
    
    norm=(voxel_distances- min_dist) / (max_dist - min_dist)
    mean=np.mean(norm)
    
    norm_up=[1 if value > 0.6  else value for value in norm]

    # fil_norm=create_filter_and_stretch_histogram(norm, 50)
    
    colors[:,1]=norm_up

    ptCloud.colors = o3d.utility.Vector3dVector(colors)
    return ptCloud, norm_up



def split_point_cloud(pcd, batch_size):
    """
    Splits the point cloud into smaller batches.

    Args:
    - pcd: open3d.geometry.PointCloud
    - batch_size: int, number of points in each batch

    Returns:
    - batches: list of open3d.geometry.PointCloud
    """
    pcd_np = np.asarray(pcd.points)
    batches = []
    for i in range(0, len(pcd_np), batch_size):
        batch_pcd = o3d.geometry.PointCloud()
        batch_pcd.points = o3d.utility.Vector3dVector(pcd_np[i:i+batch_size])
        batches.append(batch_pcd)

    print('number of batch:',len(batches))
    return batches




def sample_pcd(pcd, filter_type, n_samples, scale_min_dist, scale_max_dist, max_neighbor, var, method = "prob"):
    """ sample_pcd
    Args:
        pcd: open3d.geometry.PointCloud

        var: float
            given variance. Set to None if use data's variance

    Returns:
        pcd_sampled: open3d.geometry.PointCloud
    """

    assert method in ['prob', 'topk']
    processed=[] 
    if len(pcd.points)>1000000:
        batch_size=1000000
  
        
        for i in range(0, len(pcd.points), batch_size):

            ind=np.arange(i,min(i+batch_size,len(pcd.points)))
            batch_pcd= pcd.select_by_index(ind)

            scores_pt,scores,dis = compute_scores_from_points(batch_pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var)

            processed.extend(scores_pt)


    else:
        scores_pt,scores,dis = compute_scores_from_points(pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var)

        processed.extend(scores_pt)



    pcd_af = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scores_pt))
    # o3d.visualization.draw_geometries([pcd_af])

    pt, proc=paint_points(pcd,scores)

    o3d.visualization.draw_geometries([pt, pcd_af])

    # o3d.io.write_point_cloud('u_sco.ply',pt)



    
    # pcd_sampled = sample_points(pcd, proc, n_samples, method)
    #pcd_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sampled))

    # return pcd_sampled
    return pcd_af











def point_cloud_simplification(pcd_np,L, A, alpha, lambda_, num_points):
    """
    Simplify a point cloud based on feature preservation and density uniformity.

    Args:
    LX (np.ndarray): The product of the graph Laplacian and point cloud, shape (N, 3).
    A (csr_matrix): The adjacency matrix of the k-NN graph, shape (N, N).
    alpha (float): Simplification rate.
    lambda_ (float): Balance factor between feature loss and uniformity loss.
    num_points (int): Number of points in the original point cloud (N).

    Returns:
    np.ndarray: The simplified point cloud, shape (M, 3), where M < N.
    """
    LX=L@pcd_np
    # Objective function
    def objective(psi):
        # Ensure psi is a diagonal matrix
        Psi = diags(psi)
        # Calculate the feature loss term
        feature_loss = np.linalg.norm(Psi.dot(LX) - LX, ord=2)**2
        # Calculate the uniformity loss term
        uniformity_loss = np.linalg.norm(A.dot(Psi) - alpha * A.sum(axis=1), ord=2)**2
        # Combine the losses
        return feature_loss + lambda_ * uniformity_loss

    # Constraints
    cons = ({'type': 'eq', 'fun': lambda psi: np.sum(psi) - alpha * num_points})
    
    # Initial guess (relaxed)
    psi0 = np.full(num_points, alpha)

    # Solve the relaxed problem
    res = minimize(objective, psi0, constraints=cons, method='L-BFGS-B', options={'maxiter': 10})

    if not res.success:
        raise ValueError("Optimization failed: " + res.message)

    # Apply a heuristic to binarize the result
    # This is a placeholder for a more sophisticated binarization approach
    psi_binary = (res.x >= 0.5).astype(int)

    # Return the indices of the points to keep
    indices_to_keep = np.where(psi_binary == 1)[0]

    simplified_pcd = pcd_np[indices_to_keep]

    return indices_to_keep

# Example usage:

'''

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve

def opti(pcd_np, L, alpha=0.01, num_iter=5):

    # Parameters
    alpha = 1e-3  # Learning rate, might need tuning
    gamma = 1e-2  # Smoothness term weight, might need tuning
    iterations = 1000  # Number of iterations, might need to be increased based on convergence

    # Your data
    noisy_points = csr_matrix(pcd_np)  # Assuming this is an (N x 3) array of your noisy point cloud
    

    # Initialization
    X = noisy_points.toarray()  # Initial guess for the positions

    # Gradient Descent
    for i in tqdm(range(iterations)):
        # Compute the gradient

            # Compute the gradient using sparse matrix operations
        LX = L.dot(X)

        grad = 2 * (X - csr_matrix(noisy_points)) + 2 * gamma * (L.T.dot(LX))
    
        # gradient = 2 * (X - noisy_points.toarray()) + 2 * gamma * (L.T.dot(LX))
        
        # Update X
        X = X - alpha * grad

        # gradient = 2 * (X - noisy_points) + 2 * gamma * np.dot(L, np.dot(L, X))
        
        # # Update X
        # X = X - alpha * gradient
        
        # Optional: Check for convergence (e.g., change in cost function is below a threshold)

    # The result is in X, which should be the denoised point cloud.
    denoised_points = X
    return denoised_points



def ADMM(pcd_np, L, alpha=0.01, num_iter=5):

    # Parameters initialization
    gamma = 1e-2  # This enforces the smoothness constraint
    rho = 1.0     # Penalty parameter for ADMM, may need to be tuned

    # Assuming noisy_points is your initial noisy point cloud of shape (N, 3)
    # L is the graph Laplacian of shape (N, N)

    # Initialize variables
    X = pcd_np.copy()
    Z = pcd_np.copy()
    Y = np.zeros_like(X)   

    # # Define the update rules
    # def x_update(noisy_points, Z, Y, rho):
    #     # Solve the least squares problem:
    #     # (I + rho*L)^(-1) * (noisy_points + rho*(Z - Y))
    #     A = np.eye(noisy_points.shape[0]) + rho * L
    #     b = noisy_points + rho * (Z - Y)
    #     X_new = np.linalg.solve(A, b)
    #     return X_new


    # ADMM update functions
    def x_update(noisy_points, Z, Y, L, rho):
        # X-update involves solving a linear system to minimize the quadratic objective.
        # (I + rho * L.T.dot(L)) * X = noisy_points + rho * (L.dot(Z) - Y)
        A = np.eye(L.shape[0]) + rho * L.T @ (L)
        b = noisy_points + rho * (L@Z - Y)
        X_new = np.linalg.solve(A, b)
        return X_new

    def z_update(X, Y, gamma, rho):
        # Z update is straightforward as it involves the graph Laplacian
        Z_new =  L@X  + Y
        # (gamma/rho) *
        # A more sophisticated implementation might involve a thresholding step here
        return Z_new

    def y_update(Y, X, Z):
        # Simple dual variable update rule
        Y_new = Y + L@X -Z
        return Y_new

    # ADMM iteration loop
    max_iterations = 10
    for iteration in tqdm(range(max_iterations)):
        
        X = x_update(pcd_np, Z, Y,L, rho)
        Z = z_update(X, Y, gamma, rho)
        Y = y_update(Y, X, Z)

        # Optional: Check for convergence (not implemented here)
        # You would typically check if the norm of the primal and dual residuals are below a threshold

    # The result is in X which should now be the denoised point cloud
    denoised_points = X
    return denoised_points
'''