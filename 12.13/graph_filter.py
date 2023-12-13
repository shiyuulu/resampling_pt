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

# def calculate_distances_to_local_planes(ptCloud, pcd, k=30):
#     """
#     Calculate the distance of each point in a point cloud to its local plane.
#     Local planes are determined based on the k-nearest neighbors of each point.
#     """
#     pcd_tree = o3d.geometry.KDTreeFlann(ptCloud)
#     points = np.asarray(pcd.points)
#     distances = np.zeros(len(points))

#     for i, point in tqdm(enumerate(points)):
#         # Find k-nearest neighbors
#         [kk, idx, _] = pcd_tree.search_hybrid_vector_3d(point, 0.3,k)

#         # Fit a local plane to the neighbors
#         neighbors = points[idx[1:], :]
#         if len(neighbors)>3:
           

#     return distances






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
        
        # if k>3:

        curv=compute_local_curvature(pcd_cov[i])
    
        dist_square_value = np.asarray(dist_square)[0]
        # * np.exp(-(curv*curv)/0.0004)

        adj_matrix[i, idx] = np.exp(-dist_square_value / radius**2) 

        # adj_matrix[i, idx] = dist_square
        # *np.float16( np.exp(-(curv*curv)/0.000006))

        # +np.exp((-k*k)/100)
#        dis.append( dist_square[1])

     
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

    # row_sum = np.array(W.sum(axis=1)).flatten()  # Sum of each row in W
    # D = csr_matrix((row_sum, (range(N), range(N))), shape=(N, N))
    # return D

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize

def laplacian_smoothing(pcd_np, L, alpha=0.01, num_iter=5):
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

    # Objective function: Minimize x^T L x + alpha * ||x - x0||^2
    # where x0 is the original point cloud
    def objective(x):
        x = x.reshape(N, 3)
        return np.trace(x.T @ L @ x) + alpha * np.sum((x - pcd_np)**2)

    # Initial guess
    x0 = pcd_np.flatten()

    # Optimize
    res = minimize(objective, x0, method='BFGS', options={'maxiter': num_iter})
    
    
    if res.success:
        message = "Optimization successful."
    else:
        message = "Optimization reached the maximum number of iterations."

    return res.x.reshape(N, 3)
    # if res.success:
    #     return res.x.reshape(N, 3)
    # else:
    #     raise ValueError("Optimization failed: " + res.message)
# from scipy.sparse.linalg import lsqr

# def laplacian_smoothing(pcd_np, L, alpha=0.1, num_iter=30):
#     """
#     Apply Laplacian smoothing to a point cloud.

#     Args:
#     pcd_np (np.ndarray): The original point cloud, shape (N, 3).
#     L (np.ndarray): The graph Laplacian, shape (N, N).
#     alpha (float): Regularization parameter.
#     num_iter (int): Number of iterations for the optimization.

#     Returns:
#     np.ndarray: The smoothed point cloud, shape (N, 3).
#     """
#     N=pcd_np.shape[0]
    
#      # U matrix (Identity in this case as we're considering the entire point cloud)
#     U = np.eye(N)

#     # Optimize each dimension independently
#     X_denoised = np.zeros_like(pcd_np)
#     for c in range(pcd_np.shape[1]):
#         # Solving the least-squares problem
#         # Minimizing: lambda_val * ||X - X_denoised||^2 + ||Laplacian * X_denoised||^2
#         t = np.ravel(U.T @ L @ pcd_np[:, c])
#         C = U.T @ L @ t + alpha * pcd_np[:, c]
#         B = U.T @ L @ U + alpha * np.eye(N)
#         X_denoised[:, c] = lsqr(B, C)[0]
#     # if res.success:
#     #     message = "Optimization successful."
#     # else:
#     #     message = "Optimization reached the maximum number of iterations."

#     return X_denoised

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
    n_samples=int(0.3*len(pcd_np))

    n_sam=int(0.3*len(pcd_np))
    # Sort valid indices by scores in descending order and select top n_samples
    sampled = np.argsort(scores)[:n_samples]

    sam=np.argsort(-scores)[:n_sam]

     # Calculate average score
    # average_score = np.mean(scores)


    # # Select points and scores below average
    # mask = scores < average_score
    filtered_scores_pt = np.vstack([scores_pt[sampled],pcd_np[sam]]) 
    # filtered_scores = scores[mask]



    print(len(scores_pt))
    print(len(scores))
    print(len(pcd_np))
    return filtered_scores_pt, scores

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



import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh

def apply_haar_like_lowpass_filter(A, X):
    # Assuming A is in CSR format and is the adjacency matrix of the graph.
    
    # Compute the largest eigenvalue of A using a sparse eigenvalue solver.
    largest_eigval = eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
     # Normalize the adjacency matrix by the largest eigenvalue
    
    # Apply the low-pass filter directly to the signal
    # h_LP(X) = X + (1/largest_eigval) * A * X
    filtered_signal = X + (1.0 / largest_eigval) * (A @ X)
    
    return filtered_signal



def graph_fourier_transform(L, signal, k=None):
    if k is None:
        k = min(L.shape[0] - 2, signal.size - 1)
    eigenvalues, eigenvectors = eigsh(L, which='LM', k=100)
    transformed_signal = eigenvectors.T @ signal

    #   # Convert to dense matrix if it's sparse
    # L_dense = L.toarray()
    # eigenvalues, eigenvectors = eigh(L_dense)
    # transformed_signal = eigenvectors.T @ signal
    # eigenvalues, eigenvectors = eigsh(L, which='LM', k=signal.size-1)
    # transformed_signal = eigenvectors.T @ signal
    return transformed_signal, eigenvalues, eigenvectors

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

    radius = 0.6

    pcd.estimate_covariances( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbor))
    pcd_cov=pcd.covariances

    pcd.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbor))
    pcd_nm=np.asarray(pcd.normals)

    # compute_resolution(kd_tree, pcd_np, scale_min_dist, scale_max_dist)

    W,dis = compute_adjacency_matrix(kd_tree, pcd_np,pcd_cov, radius, max_neighbor, var)
    D = compute_D(W) 

    # transformed_signal=apply_haar_like_lowpass_filter(W,pcd_np)

    
    # L = compute_graph_laplacian(W)

    # transformed_signal, eigenvalues, eigenvectors = graph_fourier_transform(L, pcd_np)
    # plot_signal_distribution(transformed_signal, eigenvalues)


    if filter_type == "all":
        F = scipy.sparse.coo_matrix((N, N))
        F.setdiag(1)

    elif filter_type == "high":
      
        F=D-W
        # I = scipy.sparse.coo_matrix((N, N))
        # I.setdiag(1)
        # row, col, data = D.row, D.col, D.data
        # # Assuming 'data' is your numpy array
        # data = np.array(data)  # Replace with your actual data

        # # # Replace zero elements with 10
        # # data[data == 0] = 100
        # # data_inv = 1 / data
        # # data_inv[np.isnan(data_inv)] = 0  # This step is not so reasonable. 
        # D_inv = scipy.sparse.coo_matrix((np.sqrt(data), (row, col)), shape=(N, N))
        # F = D_inv @ W  
        # # F = I - A

        # Compute the degree matrix D
        row, col, data = D.row, D.col, D.data
        data = np.array(data)
        data[data == 0] = 1
        data_inv = 1/data
        # Calculate D^-1
        #D_inv = inv(D)
        D_inv =  scipy.sparse.coo_matrix((data_inv, (row, col)), shape=(N, N))

        # Apply the filter: F = D^-1 * W
        FN = D_inv @ W

    #  if filter_type == "low":
    #     # Compute the degree matrix D
    #     D = compute_D(W)

    #     # Calculate D^-1
    #     D_inv = inv(D)

    #     # Apply the filter: F = D^-1 * W
    #     F = D_inv.dot(W)


    # else:
    #     raise("Not implemented")

    scores_pt,scores = apply_filter(pcd_np, F,FN)

    # scores_pt=laplacian_smoothing(pcd_np, F)
    pcd_af = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scores_pt))

    pcd.paint_uniform_color([1,0,0])

    pcd_af.paint_uniform_color([0,1,0])

    

    o3d.visualization.draw_geometries([pcd_af,pcd])
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

    # # Create histogram bins
    # counts, bin_edges = np.histogram(data, bins=256)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # # Filter out bins with fewer elements than the threshold
    # filtered_bin_centers = [center for center, count in zip(bin_centers, counts) if count >= threshold]

    # if not filtered_bin_centers:
    #     return [], []

    # # Stretching the histogram
    # min_value = min(filtered_bin_centers)
    # max_value = max(filtered_bin_centers)
    # stretched_data = [(value - min_value) / (max_value - min_value) * 255 for value in filtered_bin_centers]

    # return stretched_data, counts

 
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
        # batches=split_point_cloud(pcd,batch_size=1000000)
        # pcd_np = np.asarray(pcd.points)
        # batches = []
        
        for i in range(0, len(pcd.points), batch_size):
            # batch_pcd = o3d.geometry.PointCloud()
            ind=np.arange(i,min(i+batch_size,len(pcd.points)))
            batch_pcd= pcd.select_by_index(ind)
            #batches.append(batch_pcd)
            scores_pt,scores,dis = compute_scores_from_points(batch_pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var)
            # processed.extend(np.multiply(scores,dis))
            processed.extend(scores_pt)
            # processed.extend(dis)

    else:
        scores_pt,scores,dis = compute_scores_from_points(pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var)
        # processed.extend(np.multiply(scores,dis))
        processed.extend(scores_pt)
        # processed.extend(dis)

        # processed=[] 


    pcd_af = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scores_pt))
    o3d.visualization.draw_geometries([pcd_af])

    pt, proc=paint_points(pcd,scores)

    o3d.visualization.draw_geometries([pt])

    # o3d.io.write_point_cloud('u_sco.ply',pt)



    
    # pcd_sampled = sample_points(pcd, proc, n_samples, method)
    #pcd_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sampled))

    # return pcd_sampled
    return pcd_af