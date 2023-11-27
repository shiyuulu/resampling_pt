import numpy as np 
import scipy.sparse
from collections import OrderedDict
import open3d as o3d

def sample_points(points, scores, n_samples, method):
    if method == 'prob':
        return sample_points_prob(points, scores, n_samples)
    elif method == 'topk':
        return sample_points_topk(points, scores, n_samples)

def sample_points_topk(points, scores, n_samples):
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
    # Filter out points with scores equal to 1

      
    # # Plot original histogram
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.hist(filtered_scores, bins=256, range=(0, 1), color='blue', alpha=0.7)
    # plt.title("Original Histogram")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # # Determine which bins to keep
    scores= np.array(scores)
    # Find indices where scores are less than 1
    valid_indices = np.where(scores < 1)[0]

    # Sort these indices by their corresponding scores, in descending order
    sorted_valid_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]

    # Select the top k indices, respecting the original order
    top_k_indices = sorted_valid_indices[:n_samples]

    # Select the corresponding points
    points_sampled = points[top_k_indices]


    # 
    # valid_indices = np.where(scores !=1)[0]
    # filtered_scores = scores[valid_indices]

    # top_k = np.argsort(scores)

    

    # ids_sampled = top_k[::-1][:n_samples]
    # points_sampled = points[ids_sampled]

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
    scores = scores / np.sum(scores)
    ids_sampled = np.random.choice(
        points.shape[0], n_samples, replace=False, p=scores)
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


import numpy as np
from scipy.spatial import cKDTree

def point_to_point_distance(pc1, pc2):
    # Create KDTree for efficient nearest neighbor search
    tree = cKDTree(pc2)
    # Find the nearest neighbor in pc2 for each point in pc1
    distances, _ = tree.query(pc1, k=1)
    # Calculate the root mean square of these distances
    rms = np.sqrt(np.mean(np.square(distances)))
    return rms

def calculate_bandwidth(point_cloud):
    # Calculate the bandwidth of the point cloud
    min_bound = np.min(point_cloud, axis=0)
    max_bound = np.max(point_cloud, axis=0)
    bandwidth = np.max(max_bound - min_bound)
    return bandwidth

def calculate_symmetric_distance(pc1, pc2):
    # Calculate point-to-point distance from pc1 to pc2 and vice versa
    distance_pc1_to_pc2 = point_to_point_distance(pc1, pc2)
    distance_pc2_to_pc1 = point_to_point_distance(pc2, pc1)
    # Symmetric distance is the maximum of these two distances
    dsym_rms = max(distance_pc1_to_pc2, distance_pc2_to_pc1)
    return dsym_rms


def calculate_psnr_geom(bandwidth, symmetric_distance):
    # Calculate the Geometry PSNR Ratio
    psnr_geom = 10 * np.log10(bandwidth**2 / symmetric_distance**2)
    return psnr_geom



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
    
   
    
    for i in tqdm(range(N)):
        [k, idx, dist_square] = kd_tree.search_hybrid_vector_3d(pcd_np[i], radius, max_neighbor)
        #[k, idx, dist_square] = kd_tree.search_radius_vector_3d(pcd_np[i], radius)
        # if k>1:
        x=compute_local_curvature(pcd_cov[i])
        adj_matrix[i,idx] = dist_square *np.exp(-x*x)* np.exp(-k*k/30)
        #adj_matrix[i,idx] = dist_square
        # else:
        #     adj_matrix[i,idx] = 0
    
    adj_matrix = adj_matrix.tocoo()
    row, col, dist_square = adj_matrix.row, adj_matrix.col, adj_matrix.data

    dist = np.sqrt(dist_square)
    data_var = var if var is not None else np.var(dist)
    print(f"var: {data_var}")
    data_new = np.exp(- dist_square / data_var)
    data_new = np.exp(- dist_square )
    adj_matrix_new = scipy.sparse.coo_matrix((data_new, (row, col)), shape=(N,N))

    return adj_matrix_new


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




def apply_filter(pcd_np, F):
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
    scores = F @ pcd_np #  (N, 3).  X_i - sum( weight_ij * neighbor(X_i)_j )
    scores = np.linalg.norm(scores, ord=2, axis=1)  # [N], L2 distance
    #scores = scores ** 2  # [N], L2 distance square 

    return scores



def compute_scores_from_points(pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var):
    """ compute scores for each point.

    """
  

    pcd_np = np.asarray(pcd.points)
    N = pcd_np.shape[0]
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    radius = 1

    pcd.estimate_covariances( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_neighbor))
    pcd_cov=pcd.covariances

    # compute_resolution(kd_tree, pcd_np, scale_min_dist, scale_max_dist)

    W = compute_adjacency_matrix(kd_tree, pcd_np,pcd_cov, radius, max_neighbor, var)
    D = compute_D(W) 

    if filter_type == "all":
        F = scipy.sparse.coo_matrix((N, N))
        F.setdiag(1)

    elif filter_type == "high":
      
        F=D-W
        # I = scipy.sparse.coo_matrix((N, N))
        # I.setdiag(1)
        # row, col, data = D.row, D.col, D.data
        # data_inv = 1 / data
        # data_inv[np.isnan(data_inv)] = 0  # This step is not so reasonable. 
        # D_inv = scipy.sparse.coo_matrix((data_inv, (row, col)), shape=(N, N))
        # A = D_inv @ W  
        # F = I - A

    else:
        raise("Not implemented")

    scores = apply_filter(pcd_np, F)

    return scores


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
    
    norm_up=[1 if value > 0.45 else value for value in norm]

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
    if len(pcd.points)>2000000:
        batch_size=2000000
        # batches=split_point_cloud(pcd,batch_size=1000000)
        # pcd_np = np.asarray(pcd.points)
        # batches = []
        
        for i in range(0, len(pcd.points), batch_size):
            # batch_pcd = o3d.geometry.PointCloud()
            ind=np.arange(i,min(i+batch_size,len(pcd.points)))
            batch_pcd= pcd.select_by_index(ind)
            #batches.append(batch_pcd)
            scores = compute_scores_from_points(batch_pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var)
            processed.extend(scores)
    else:
        scores = compute_scores_from_points(pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var)
        processed.extend(scores)
        # processed=[] 



    pt, proc=paint_points(pcd,processed)

    o3d.visualization.draw_geometries([pt])

    o3d.io.write_point_cloud('u_curv_density.ply',pt)


    pcd_np = np.asarray(pcd.points)
    points_sampled = sample_points(pcd_np, proc, n_samples, method)
    pcd_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sampled))

    return pcd_sampled
