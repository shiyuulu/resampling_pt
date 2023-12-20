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
    # smallest_eigenvalue = eigenvalues[sorted_indices[0]]

    # return smallest_eigenvalue / (eigenvalues.sum())
    return eigenvalues[sorted_indices[1]]- eigenvalues[sorted_indices[2]]/ eigenvalues[sorted_indices[0]]

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
        
        #curv=compute_local_curvature(pcd_cov[i])
    
        dist_square_value = np.asarray(dist_square)[0]
        # * np.exp(-(curv*curv)/0.0004)
        #dis.append(curv)

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

def laplacian_smoothing(pcd_np, L, NF, F, gamma=0.15, f_gamma=0.96, num_iter=800):
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

    

    def objective(x):
        x = x.reshape(N, 3)
        # Sx=S@x
        laplacian_term =  gamma *np.sum((x.T@L@ x)**2)
        
        # np.sum(L.dot(x)* x  )     #laplacan loss
        #np.sum((x - pcd_np)**2)  #distance loss
        # np.sum((x - L.dot(x))**2)  #feature loss
        #np.sum(x.T@L@ x)            # #total vari
        # np.sum((L.dot(x))**2)       #smooth
        regularization_term = (1-gamma)*(f_gamma*np.linalg.norm(F @ (x - pcd_np))**2+(1-f_gamma)*np.sum(NF @ (x - pcd_np)**2)+0.8*np.sum((x - pcd_np)**2))

        return laplacian_term + regularization_term

    def gradient(x):
        x = x.reshape(N, 3)
        # Sx=S@x
        laplacian_grad =  gamma * 2 *L.dot(x)       
        # 2 * L.dot(x)              #laplacan loss
        # (x - pcd_np)                 #distance loss
        # L.T.dot(L.dot(x) - x)     #feature loss
        # L.dot(x)                     #total vari
        # *L.T.dot(L.dot(x))            #smooth
        regularization_grad =(1-gamma) *2 *(f_gamma*F @ (x - pcd_np)+ (1-f_gamma)*NF @(x - pcd_np)+ 0.8*(x - pcd_np))
        return (laplacian_grad + regularization_grad).flatten()

    # @works well:
    # smooth/ total ari +distance loss regularization.  gamma= 0.1

    # Initial guess
    x0 = pcd_np.flatten()

    # Optimize
    res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', options={'maxiter': num_iter})
    
    
    #print(f"Number of calls to Simulator instance {res.num_calls}")
    if res.success:
        message = "Optimization successful."
    else:
        message = "Optimization reached the maximum number of iterations."

    # pt=np.array(res.x.reshape(N, 3))
    # s_pt=np.array(res.x.reshape(N, 3))[features==0]
    # f_pt=pcd_np[features==1]

    # pt=np.vstack([s_pt,f_pt])


    return res.x.reshape(N, 3)
   


def apply_filter(pcd_np, F,dis):
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
    scores=F @  pcd_np
    scores = np.linalg.norm(scores, ord=2, axis=1)  # [N], L2 distance
    scores = scores ** 2  # [N], L2 distance square 

    scores=np.asarray(scores)  
    cur= np.array(dis)
    thre=np.median(cur)
    # cur_ind=np.where (cur > thre)[0]
    

    n_sam=int(0.1*len(pcd_np))

     # Sort scores in descending order and get indices
    sorted_indices = np.argsort(-scores)

    # # Select top indices where dis condition is met
    # sam = [idx for idx in sorted_indices if idx in cur_ind][:n_sam]
    sam = sorted_indices[int(0.001*len(pcd_np)):n_sam+int(0.001*len(pcd_np))]

    # filtered_scores_pt = np.vstack([scores_pt[sampled],pcd_np[sam]]) 
    filtered_scores_pt = pcd_np[sam]
    # filtered_scores = scores[mask]

    # Initialize the binary array with zeros
    features = np.zeros(len(pcd_np), dtype=int)  # N should be the total number of points in your point cloud

    # Set the entries at the feature indices to 1
    features[sam] = 1

    # print(len(scores))
    # print(len(filtered_scores_pt))

    return filtered_scores_pt, scores,features



from scipy.spatial.transform import Rotation as RR

def rotation_matrix_to_angle(R):
    # Convert rotation matrix to axis-angle representation and return the angle
    # r = RR.from_matrix(R)
    # axis_angle = r.as_rotvec()
    # angle = np.linalg.norm(axis_angle)  # Calculate 
    # return angle
    # Assuming the rotation matrix is in SO(3)
    theta = np.arccos((np.trace(R) - 1) / 2)
    return theta
def calculate_errors(ground_truth_matrix, calculated_matrix):
    # Extract rotation and translation components

     # Calculate the change in transformation
    delta_T = np.dot(np.linalg.inv(calculated_matrix),ground_truth_matrix)

    # Extract the rotation and translation differences
    delta_R = delta_T[:3, :3]
    delta_t = delta_T[:3, 3]
     
    # Calculate rotation error
    trace_R = np.trace(delta_R)
    e_r = np.arccos((trace_R - 1) / 2)

     # Calculate translation error
    e_t = np.linalg.norm(delta_t)
    
    

    return np.degrees(e_r),e_t

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def fine_reg(source, target, s, t,trans_ini, gt_T):

  
    print('fine registration')
    # trans_ini = np.identity(4)

    # s_down = source.voxel_down_sample(voxel_size=0.05)
    # t_down = target.voxel_down_sample(voxel_size=0.05)


    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))

    simplify_icp = o3d.pipelines.registration.registration_icp(
    source, target, 0.1, trans_ini,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    print(simplify_icp)
    
      

    simplify_icp_p2p = o3d.pipelines.registration.registration_icp(
    source, target, 0.1, trans_ini,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    print(simplify_icp_p2p)


    # t_col=np.dot(feature_T,result_icp_c.transformation)
    t_icp_simplify=np.dot(simplify_icp.transformation,trans_ini)

    t_icp_simplify_p2p=np.dot(simplify_icp_p2p.transformation,trans_ini)

    rotation_err, translation_err = calculate_errors(gt_T, t_icp_simplify)
    print('simplified')
    print(f"Rotation Error: {rotation_err} degree")
    print(f"Translation Error: {translation_err} units")


    eva_icp_simplify=o3d.pipelines.registration.evaluate_registration(target,source.transform(t_icp_simplify), 0.2)
    print("inlier_RMSE",eva_icp_simplify.inlier_rmse)
    print("inlier_ratio",eva_icp_simplify.fitness)
    

    rotation_error, translation_error = calculate_errors(gt_T, t_icp_simplify_p2p)
    print('simplified_p2p')
    print(f"Rotation Error: {rotation_error} radians")
    print(f"Translation Error: {translation_error} units")
    eva_icp_simplify_p2p=o3d.pipelines.registration.evaluate_registration(target,source.transform(t_icp_simplify_p2p), 0.2)
    print("inlier_RMSE",eva_icp_simplify_p2p.inlier_rmse)
    print("inlier_ratio",eva_icp_simplify_p2p.fitness)



    s.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))
    t.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))
    
    Original_icp_p2p = o3d.pipelines.registration.registration_icp(
    s, t, 0.1, trans_ini,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    # ss_down = s.voxel_down_sample(voxel_size=0.05)
    # tt_down = t.voxel_down_sample(voxel_size=0.05)
    
    
    print(Original_icp_p2p)

    Original_icp = o3d.pipelines.registration.registration_icp(
    s,t, 0.1, trans_ini,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    print(Original_icp)

    # result_icp_c = o3d.pipelines.registration.registration_colored_icp(
    #     s_downc, t_downc, 0.1, current_transformation,
    #     o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                                       relative_rmse=1e-6,
    #                                                       max_iteration=100))
    
   


    
    # draw_registration_result_original_color(s, t,
    #                                         result_icp_c.transformation)

    draw_registration_result_original_color(source, target,
                                            simplify_icp.transformation)
    
    draw_registration_result_original_color(s, t,
                                            Original_icp.transformation)


    t_icp_ori_p2p=np.dot(Original_icp_p2p.transformation,trans_ini)



    t_icp_ori=np.dot(Original_icp.transformation,trans_ini)
    



   
    rotation_error, translation_error = calculate_errors(gt_T, t_icp_ori)
    print('ori')
    print(f"Rotation Error: {rotation_error} radians")
    print(f"Translation Error: {translation_error} units")
    eva_icp_ori=o3d.pipelines.registration.evaluate_registration(target,source.transform(t_icp_ori), 0.2)
    print("inlier_RMSE",eva_icp_ori.inlier_rmse)
    print("inlier_ratio",eva_icp_ori.fitness)

    rotation_error, translation_error = calculate_errors(gt_T, t_icp_ori_p2p)
    print('ori_p2p')
    print(f"Rotation Error: {rotation_error} radians")
    print(f"Translation Error: {translation_error} units")
    eva_icp_ori_p2p=o3d.pipelines.registration.evaluate_registration(target,source.transform(t_icp_ori_p2p), 0.2)
    print("inlier_RMSE",eva_icp_ori_p2p.inlier_rmse)
    print("inlier_ratio",eva_icp_ori_p2p.fitness)



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

def merge_point_clouds(*point_clouds):
    """
    Merge multiple Open3D point clouds.

    Parameters:
    - *point_clouds: Any number of Open3D point clouds.

    Returns:
    - Merged Open3D point cloud.
    """
    
    # Check if there's at least one point cloud
    if not point_clouds:
        raise ValueError("At least one point cloud must be provided.")
    
    # Initialize lists to store points and colors
    all_points = []
    all_colors = []
    
    for pcd in point_clouds:

        all_points.append(np.asarray(pcd.points))
        
        # If point cloud has colors, store them
        if pcd.has_colors():
            all_colors.append(np.asarray(pcd.colors))

    # Concatenate all points and colors
    merged_points = np.vstack(all_points)

    # Check if all point clouds had colors
    if len(all_colors) == len(point_clouds):
        merged_colors = np.vstack(all_colors)
    else:
        merged_colors = None

    # Create a new point cloud with merged points
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    
    if merged_colors is not None:
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    return merged_pcd



def compute_scores_from_points(pcd, kd_tree, radius, gamma, filter_type,  max_neighbor, var, voxel_size = 0.02 ):
    """ compute scores for each point.

    """
  

    pcd_np = np.asarray(pcd.points)
    N = pcd_np.shape[0]
   
    

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
     
    # scores_pt,scores = apply_filter(pcd_np, F,FN)
    sample_pt,scores, feature = apply_filter(pcd_np, F,dis)

    NF_mask = diags(feature)
    F_mask=diags(feature)


    scores_pt= laplacian_smoothing(pcd_np, F,NF_mask,F_mask, gamma=0.1,f_gamma=0.9)

    pcd_fea = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sample_pt))

    pcd_af = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scores_pt))

   
    if len(pcd.colors):
        pcd_fea.colors=o3d.utility.Vector3dVector(np.array(pcd.colors)[feature==1])
        pcd_af.colors=o3d.utility.Vector3dVector(pcd.colors)

        
     # for example, 5cm
    # Apply voxel downsampling
    downsampled_pcd = pcd_af.voxel_down_sample(voxel_size)



    pcd_simplify=merge_point_clouds(pcd_fea.voxel_down_sample(0.5*voxel_size),downsampled_pcd)


    return pcd_simplify


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

import copy


def sample_pcd(pcd, filter_type,voxel_size, radius, gamma, max_neighbor, var, method = "prob"):
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
    scores_p=[]
   
    if len(pcd.points)>2000000:
        batch_size=2000000

        # pcd_sort=o3d.geometry.PointCloud()
        # ptlist=np.array(pcd.points)
        # clist=np.array(pcd.colors)
        # ptsort=ptlist[np.argsort(ptlist[:,2])]
        # csort=clist[np.argsort(ptlist[:,2])]
        # pcd_sort.points= o3d.utility.Vector3dVector(ptsort)
        # pcd_sort.colors= o3d.utility.Vector3dVector(csort)
        # o3d.visualization.draw_geometries([pcd_sort])

        for i in range(0, len(pcd.points), batch_size):

            ind=np.arange(i,min(i+batch_size,len(pcd.points)))

            batch_pcd= pcd.select_by_index(ind)

            kd_tree = o3d.geometry.KDTreeFlann(batch_pcd)

            pcd_sim = compute_scores_from_points(batch_pcd, kd_tree, radius, gamma, filter_type,  max_neighbor, var, voxel_size)

            processed.append(pcd_sim)
            # scores_p.extend(scores)
        pcd_simplify=merge_point_clouds(*processed)
        # o3d.visualization.draw_geometries([pcd_simplify])
        # pcd_af = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(processed))
        # pcd_af.colors=o3d.utility.Vector3dVector( csort)
        # o3d.visualization.draw_geometries([pcd_af])
        # pt, proc=paint_points(pcd,scores_p)


    else:
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        pcd_simplify = compute_scores_from_points(pcd,kd_tree, radius, gamma, filter_type, max_neighbor, var, voxel_size)

        # processed.extend(scores_pt)
        # scores_p.extend(scores)
        # pcd_af = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scores_pt))
        # pcd_af.colors=o3d.utility.Vector3dVector( o3d.utility.Vector3dVector(pcd.colors))
        # o3d.visualization.draw_geometries([pcd_simplify])
        # pt, proc=paint_points(pcd,scores)


    
   
    # pcd_af.colors=o3d.utility.Vector3dVector(clist)
    # o3d.visualization.draw_geometries([pcd_af])


   

    # o3d.visualization.draw_geometries([pt.translate([-50,0,0]), pcd_af])

    # o3d.io.write_point_cloud('MM_smoothed.ply',pcd_af)



    
    # pcd_sampled = sample_points(pcd, proc, n_samples, method)
    #pcd_sampled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_sampled))

    # return pcd_sampled
    return pcd_simplify











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