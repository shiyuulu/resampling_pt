import numpy as np 
import scipy.sparse
from collections import OrderedDict
import open3d as o3d




def subsample_point_cloud(point_cloud, subsample_rate):
    """
    Subsamples a 3D point cloud using a voxel grid approach, based on a given subsample rate.

    :param point_cloud: NumPy ndarray of shape (N, 3) representing the point cloud.
    :param subsample_rate: Desired rate of subsampling (0 < subsample_rate <= 1).
    :return: Subsampled point cloud as a NumPy ndarray.
    """

    if not (0 < subsample_rate <= 1):
        raise ValueError("Subsample rate must be between 0 and 1.")

    # Total number of points desired in the subsampled cloud
    target_num_points = int(len(point_cloud) * subsample_rate)

    # Approximate the voxel size based on the desired number of points
    total_volume = np.prod(np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0))
    voxel_volume = total_volume / target_num_points
    voxel_size = voxel_volume ** (1/3)  # Cube root to find the edge length of the voxel

    # Define the voxel grid
    min_bound = np.min(point_cloud, axis=0)  # Minimum coordinates
    max_bound = np.max(point_cloud, axis=0)  # Maximum coordinates

    # Number of voxels in each dimension
    dims = ((max_bound - min_bound) / voxel_size).astype(int) + 1

    # Assign each point to a voxel
    indices = ((point_cloud - min_bound) / voxel_size).astype(int)

    # Dictionary to hold points for each voxel
    voxels = {}
    for point, index in zip(point_cloud, indices):
        voxel_key = tuple(index)
        if voxel_key not in voxels:
            voxels[voxel_key] = []
        voxels[voxel_key].append(point)

    # Subsample one point per voxel
    subsampled_points = []
    for voxel in voxels.values():
        subsampled_points.append(voxel[np.random.randint(len(voxel))])

    return np.array(subsampled_points)



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
   
    # Find indices where scores are not 1
    scores=np.asarray(scores)
    valid_indices = np.where(scores != 1)[0]

    # Sort valid indices by scores in descending order and select top n_samples
    top_k_indices = np.argsort(-scores[valid_indices])[:n_samples]

    # Select the corresponding points
    points_sampled = points[valid_indices[top_k_indices]]

    # Sort valid indices by scores in descending order and select top n_samples
    end_k_indices = np.argsort(scores[valid_indices])[:2*n_samples]

    points_sampled=np.vstack([points_sampled,points[valid_indices[end_k_indices]]])
    # rest=subsample_point_cloud(points[valid_indices[top_k_indices]])

    #pcd_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sampled))



    


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
    
    dis=[]
    
    for i in tqdm(range(N)):
        [k, idx, dist_square] = kd_tree.search_hybrid_vector_3d(pcd_np[i], radius, max_neighbor)
       
        dist_square_value = np.asarray(dist_square)[0]
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
    scores = scores ** 2  # [N], L2 distance square 

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

    W,dis = compute_adjacency_matrix(kd_tree, pcd_np,pcd_cov, radius, max_neighbor, var)
    D = compute_D(W) 

    if filter_type == "all":
        F = scipy.sparse.coo_matrix((N, N))
        F.setdiag(1)

    elif filter_type == "high":
      
        F=D-W
    #     I = scipy.sparse.coo_matrix((N, N))
    #     I.setdiag(1)
    #     row, col, data = D.row, D.col, D.data
    #     # Assuming 'data' is your numpy array
    #     data = np.array(data)  # Replace with your actual data

    #     # Replace zero elements with 10
    #     data_inv = 1 / np.where(data !=0, np.sqrt(data), np.inf)
        
    #     # # data[data == 0] = 1
    #     # data_inv = 1 / data
    #    # data_inv[np.isnan(data_inv)] = 0  # This step is not so reasonable. 
    #     D_inv = scipy.sparse.coo_matrix((data_inv, (row, col)), shape=(N, N))
    #     # A = D_inv @ W  
    #     A = D_inv @ F @ D_inv
        
    #     F =  A
        # F = I - A

    else:
        raise("Not implemented")

    scores = apply_filter(pcd_np, F)

    return scores, dis


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




def sample_pcd(pcd, filter_type, rate, scale_min_dist, scale_max_dist, max_neighbor, var, method = "prob"):
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
           
            scores,dis = compute_scores_from_points(batch_pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var)
            # processed.extend(np.multiply(scores,dis))
            processed.extend(scores)
            # processed.extend(dis)

    else:
        scores,dis = compute_scores_from_points(pcd, filter_type, scale_min_dist, scale_max_dist, max_neighbor, var)
        # processed.extend(np.multiply(scores,dis))
        processed.extend(scores)
        # processed.extend(dis)

    

    pt, proc=paint_points(pcd,processed)

   # o3d.visualization.draw_geometries([pt])
   

    # o3d.io.write_point_cloud('sfm_c.ply',pt)


    pcd_np = np.asarray(pcd.points)
    n_samples=int(len(pcd_np)*rate)
    points_sampled = sample_points(pcd_np, proc, n_samples, method)
    pcd_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sampled))

    o3d.visualization.draw_geometries([pcd_sampled])
    


    return pcd_sampled,pt

import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

import copy

def match_histograms(source_colors, target_colors):
    # Extracting the green channel values
    source_greens = np.array([color[1] for color in source_colors])
    target_greens = np.array([color[1] for color in target_colors])

    # Matching histograms
    matched_greens = exposure.match_histograms(source_greens, target_greens)

    # Reconstructing the adjusted RGB colors
    adjusted_colors = [[0, green, 0] for green in matched_greens]

    return adjusted_colors,source_greens,target_greens



def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
                                      

def plot_histograms(before_c, after_c, title):
    before = [color[1] for color in before_c]
    after = [color[1] for color in after_c]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(before, bins=256, color='green', alpha=0.7)
    plt.title(f'Before {title}')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(after, bins=256, color='green', alpha=0.7)
    plt.title(f'After {title}')
    plt.xlabel('Intensity')

    plt.show()

from scipy.spatial.transform import Rotation as RR

def rotation_matrix_to_angle(R):
    # Convert rotation matrix to axis-angle representation and return the angle
    r = RR.from_matrix(R)
    axis_angle = r.as_rotvec()
    angle = np.linalg.norm(axis_angle)  # Calculate 
    return angle
def calculate_errors(ground_truth_matrix, calculated_matrix):
    # Extract rotation and translation components
    R_gt = ground_truth_matrix[:3, :3]
    t_gt = ground_truth_matrix[:3, 3]
    R_calc = calculated_matrix[:3, :3]
    t_calc = calculated_matrix[:3, 3]

    # Calculate rotation error
    rotation_diff = np.dot(R_calc, np.linalg.inv(R_gt))
    rotation_error = rotation_matrix_to_angle(rotation_diff)

    # Calculate translation error
    translation_error = np.linalg.norm(t_calc - t_gt)

    return rotation_error, translation_error


def fine_reg(source, target, feature_T, gt_T, s, t):

    s_color= np.asarray(s.colors)
    t_color= np.asarray(t.colors)

    # # u_match=match_histograms(u_color,m_color)
   
    # # u_match= np.asarray(equalize_color_histogram(u_color))

    
    # # # Plotting the histograms
    # t_match,m_f,u_f=match_histograms(t_color,s_color)
    # #plot_histograms(t_color, t_match, "Source Histogram Matching")
    # t_temp = copy.deepcopy(target)
    # t_temp.colors=o3d.utility.Vector3dVector(t_match)


     # # Plotting the histograms
    if len(s_color) and len(t_color):
        s_match,m_f,u_f=match_histograms(s_color,t_color)
        #plot_histograms(t_color, t_match, "Source Histogram Matching")
        s_temp = copy.deepcopy(s)
        s_temp.colors=o3d.utility.Vector3dVector(s_match)

        o3d.visualization.draw_geometries([s_temp,t])
        # o3d.visualization.draw_geometries([pcd_orig])
        s_downc = s_temp.voxel_down_sample(voxel_size=0.05)
        t_downc = t.voxel_down_sample(voxel_size=0.05)
        s_downc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))
        t_downc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))

        # m_c=copy.deepcopy(t_temp)
        # u_c=copy.deepcopy(pcd_u)

    current_transformation = np.identity(4)

    s_down = source.voxel_down_sample(voxel_size=0.05)
    t_down = target.voxel_down_sample(voxel_size=0.05)


    s_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))
    t_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))

    result_icp = o3d.pipelines.registration.registration_icp(
    s_down, t_down, 0.1, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )


    ss_down = s.voxel_down_sample(voxel_size=0.05)
    tt_down = t.voxel_down_sample(voxel_size=0.05)


    ss_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))
    tt_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))

    result_icpp = o3d.pipelines.registration.registration_icp(
    ss_down, tt_down, 0.1, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    result_icp_c = o3d.pipelines.registration.registration_colored_icp(
        s_downc, t_downc, 0.1, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=100))
    
    print(result_icp)
    print(result_icp_c)
    draw_registration_result_original_color(s, t,
                                            result_icp_c.transformation)

    draw_registration_result_original_color(s, t,
                                            result_icp.transformation)
    
    draw_registration_result_original_color(s, t,
                                            result_icpp.transformation)

    t_col=np.dot(feature_T,result_icp_c.transformation)
    t_icp=np.dot(feature_T,result_icp.transformation)

    t_icpp=np.dot(feature_T,result_icpp.transformation)
    


    rotation_error, translation_error = calculate_errors(gt_T, t_col)
    print(f"Rotation Error: {rotation_error} radians")
    print(f"Translation Error: {translation_error} units")


    rotation_err, translation_err = calculate_errors(gt_T, t_icp)
    print(f"icpRotation Error: {rotation_err} radians")
    print(f"Translation Error: {translation_err} units")


    rotation_error, translation_error = calculate_errors(gt_T, t_icpp)
    print(f"nosubRotation Error: {rotation_error} radians")
    print(f"Translation Error: {translation_error} units")