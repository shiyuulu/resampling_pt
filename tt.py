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
    scores= np.array(scores)
    valid_indices = np.where(scores < 1)[0]
    sorted_valid_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    # Select the top k indices, respecting the original order
    top_k_indices = sorted_valid_indices[:n_samples]
    # Select the corresponding points
    points_sampled = points[top_k_indices]


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
        if len(pcd_cov[i]):
            cur=compute_local_curvature(pcd_cov[i])
            # adj_matrix[i,idx] = dist_square *np.exp(-cur*cur)   *np.exp(-cur*cur)
            m=0.2
            adj_matrix[i,idx] = dist_square *np.float16(m*np.exp(-cur*cur/0.005)+ (1-m)*k/max_neighbor)
        # np.exp(k*k)
        #adj_matrix[i,idx] = dist_square
        # else:
        #     adj_matrix[i,idx] = 0
        else:
            adj_matrix[i,idx] = 0
    
    adj_matrix = adj_matrix.tocoo()
    row, col, dist_square = adj_matrix.row, adj_matrix.col, adj_matrix.data

    # dist = np.sqrt(dist_square)
    # data_var = var if var is not None else np.var(dist)
    # print(f"var: {data_var}")
    # data_new = np.exp(- dist_square / data_var)
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
    # scores = F @ pcd_np #  (N, 3).  X_i - sum( weight_ij * neighbor(X_i)_j )
   
    # transformed = F @ pcd_np  # Step 1: Apply Laplacian transformation
    # # product = 
    # scores = np.diag(transformed @ transformed.T)
    scores= F @ pcd_np 
    scores = np.linalg.norm(scores, ord=2, axis=1)  # [N], L2 distance
    scores = scores ** 2  # [N], L2 distance square 

    return scores



def compute_scores_from_points(pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var):
    """ compute scores for each point.

    """
  

    pcd_np = np.asarray(pcd.points)
    N = pcd_np.shape[0]
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    radius =1

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
        # data_inv = 1 / np.where(data > 1, data, 1)
        # #data_inv = 1 / data
        # #data_inv[np.isnan(data_inv)] = 0  # This step is not so reasonable. 
        # D_inv = scipy.sparse.coo_matrix((data_inv, (row, col)), shape=(N, N))
        # A = D_inv @ W  
        # F = I - A

    else:
        raise("Not implemented")

    scores = apply_filter(pcd_np, F)
    return scores

 
def paint_points(ptCloud, voxel_distances):
    points = np.asarray(ptCloud.points)
    colors = np.zeros_like(points)

    # Flatten all distances to find global min and max
    all_distances =list(voxel_distances)
    min_dist, max_dist = np.min(all_distances), np.max(all_distances)


    print(max_dist)
    print(min_dist)
    mean=np.mean(all_distances) 
    var=np.var(all_distances)
    print(mean)
    print(var)
    # norm=(all_distances - min_dist) / (max_dist - min_dist)
    

    dis=[6*mean if value >6*mean else value for value in all_distances]


    min_dis, max_dis = np.min(dis), np.max(dis)

    print(max_dis)
    print(min_dis)
    print(np.mean(dis))
    

    norm=(dis- min_dis) / (max_dis - min_dis)
    norm_up=[1 if value > 0.6 else value for value in norm]
    colors[:,1]=norm

    ptCloud.colors = o3d.utility.Vector3dVector(colors)
    return ptCloud, norm



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


    pt,dis=paint_points(pcd,processed)

    o3d.visualization.draw_geometries([pt])

    o3d.io.write_point_cloud('uum.ply',pt)


    pcd_np = np.asarray(pcd.points)
    points_sampled = sample_points(pcd_np, dis, n_samples, method)
    pcd_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sampled))

    return pcd_sampled
