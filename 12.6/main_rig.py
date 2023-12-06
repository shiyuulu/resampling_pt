import open3d as o3d
from utils import read_data, draw, read_data
from graph_filter import sample_pcd, paint_points
import numpy as np
from tqdm import tqdm


def select_main(pcd, threshold=0.1):
    # Extract colors from the point cloud
    colors = np.asarray(pcd.colors)

    # Select points where the green channel is less than the specified threshold
    selected_indices = np.where(colors[:, 1] < 0.6   )[0]

    # Filter the point cloud based on the selected indices
    filtered_pcd = pcd.select_by_index(selected_indices)

    return filtered_pcd

import copy

import open3d as o3d
import numpy as np
import pyransac3d as pyrsc


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      
                                    )
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

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


# def plot_histograms(before, after):
#     plt.figure(figsize=(12, 6))
    
#     # Plotting the original histogram
#     plt.subplot(1, 2, 1)
#     plt.hist(before, bins=256, color='green', alpha=0.7)
#     plt.title('Original Green Channel Histogram')
#     plt.xlabel('Intensity')
#     plt.ylabel('Frequency')

#     # Plotting the equalized histogram
#     plt.subplot(1, 2, 2)
#     plt.hist(after, bins=256, color='green', alpha=0.7)
#     plt.title('Equalized Green Channel Histogram')
#     plt.xlabel('Intensity')

#     plt.show()


def match_histograms(source_colors, target_colors):
    # Extracting the green channel values
    source_greens = np.array([color[1] for color in source_colors])
    target_greens = np.array([color[1] for color in target_colors])

    # Matching histograms
    matched_greens = exposure.match_histograms(source_greens, target_greens)

    # Reconstructing the adjusted RGB colors
    adjusted_colors = [[0, green, 0] for green in matched_greens]

    return adjusted_colors,source_greens,target_greens


def compute_pca_and_align_centers(pcd1, pcd2):
    # Function to compute centroid and PCA of a point cloud
    def pca_point_cloud(pcd):
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        covariance_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        return centroid, eigenvectors, eigenvalues
 # Function to compute rotation matrix to align vectors v1 to v2
    def rotation_matrix_from_vectors(v1, v2):
        axis = np.cross(v1, v2)
        axis_length = np.linalg.norm(axis)
        if axis_length < 1e-8:
            # Vectors are parallel
            return np.eye(3)

        axis = axis / axis_length
        angle = np.arccos(np.dot(v1, v2))
        axis_skew = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
        rotation_matrix = np.eye(3) + np.sin(angle) * axis_skew + (1 - np.cos(angle)) * np.dot(axis_skew, axis_skew)
        return rotation_matrix
    # Load point clouds
    # pcd1 = o3d.io.read_point_cloud("path_to_first_point_cloud.ply")
    # pcd2 = o3d.io.read_point_cloud("path_to_second_point_cloud.ply")

    # Compute PCA for each point cloud
    centroid1, eigenvectors1, eigenvalues1 = pca_point_cloud(pcd1)
    centroid2, eigenvectors2, eigenvalues2 = pca_point_cloud(pcd2)

    # The main direction is the eigenvector associated with the largest eigenvalue
    main_direction1 = eigenvectors1[:, np.argmax(eigenvalues1)]
    main_direction2 = eigenvectors2[:, np.argmax(eigenvalues2)]

    # Align centroids
    translation = centroid2 - centroid1
    pcd1.translate(translation)
      # Find the rotation matrix
    rotation_matrix = rotation_matrix_from_vectors(main_direction1, main_direction2)

    # Apply the rotation to align the main directions
    pcd1.rotate(rotation_matrix, center=centroid2)


    # Return the results
    return pcd1, pcd2

# def equalize_color_histogram(colors):
#     # Extract the green channel
#     greens = np.array([color[1] for color in colors])

#     # Perform histogram equalization on the green channel
#     equalized_greens = exposure.equalize_hist(greens)

#     # Scale the equalized greens back to the original range
#     #equalized_greens = np.round(equalized_greens * 255).astype(int)

#     # Reconstruct the RGB colors
#     equalized_colors = [[0, green, 0] for green in equalized_greens]

#     return equalized_colors

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

def inver_trans(T):
        
    # Compute the inverse transformation matrix
    R = T[:3, :3]  # Extract rotation matrix
    t = T[:3, 3]   # Extract translation vector

    R_inv = R.T  # Transpose of rotation matrix for the inverse
    t_inv = -np.dot(R_inv, t)  # Inverse translation vector

    T_inv = np.identity(4)  # Initialize 4x4 identity matrix
    T_inv[:3, :3] = R_inv   # Set top-left 3x3 to rotation inverse
    T_inv[:3, 3] = t_inv   # Set top-right 3x1 to translation inverse
    return T_inv


if __name__ == "__main__":
    # path_data = '../startfrom_nov_uncer/83bm.ply'
    # path_data = 'fil_u.ply'
    path_data = 'sfm_c.ply'
    # path_data = 'cone.ply'
    path2='kin.ply'
    # path_data ='data/cone.ply'
    filter_type = "high" # or 'all'

    rate=0.1


    scale_min_dist = 3
    scale_max_dist = 10
    max_neighbor = 60
    var = None#None # float number or None
    method = 'topk' # 'topk' or 'prob'

    pcd_orig = read_data(path_data)
    pcd_u=read_data(path2)

    # pcd_orig.paint_uniform_color([0, 0, 0])

    u_color= np.asarray(pcd_u.colors)
    m_color= np.asarray(pcd_orig.colors)

    # # u_match=match_histograms(u_color,m_color)
    m_match,m_f,u_f=match_histograms(m_color,u_color)
    # # u_match= np.asarray(equalize_color_histogram(u_color))

    
    # # Plotting the histograms
    plot_histograms(m_color, m_match, "Source Histogram Matching")
    # # plot_histograms(u_match, m_color, "Target Histogram")
    m_temp = copy.deepcopy(pcd_orig)

    m_temp.colors=o3d.utility.Vector3dVector(m_match)
    o3d.visualization.draw_geometries([m_temp,pcd_u])
    # o3d.visualization.draw_geometries([pcd_orig])

    m_c=copy.deepcopy(m_temp)
    u_c=copy.deepcopy(pcd_u)
    # u_c,m_c=compute_pca_and_align_centers(pcd_u, pcd_orig)

    # result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #     pcd_u, m_c, u_f, m_f, 1,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
    #     [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(20)],
    #     o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    # u_c=pcd_u.transformation(result.transformation)

    # o3d.visualization.draw_geometries([m_c,u_c])



    orim=o3d.io.read_point_cloud("kinect.ply")
    oriu=o3d.io.read_point_cloud("sfm.ply")

    dismu=orim.compute_point_cloud_distance(oriu)

    disum=oriu.compute_point_cloud_distance(orim)
    # o3d.visualization.draw_geometries([pcd_u]) 

    # m_main=select_main(m_temp, threshold=0.1)
    # u_main=select_main(pcd_u, threshold=0.1)
    
    # o3d.visualization.draw_geometries([m_main])  # Uncomment to visualize

    # o3d.visualization.draw_geometries([u_main])  # Uncomment to visualize''

    #
    current_transformation = np.identity(4)

    downu = u_c.voxel_down_sample(voxel_size=0.01)
    downm = m_c.voxel_down_sample(voxel_size=0.01)


    downu.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))
    downm.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))

    result_icp = o3d.pipelines.registration.registration_icp(
    downu, downm, 0.1, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    


    result_icp_c = o3d.pipelines.registration.registration_colored_icp(
        downu, downm, 0.1, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=100))
    print(result_icp)
    draw_registration_result_original_color(u_c, m_c,
                                            result_icp_c.transformation)

    draw_registration_result_original_color(u_c, m_c,
                                            result_icp.transformation)

    u_temp=copy.deepcopy(oriu)

    u_temp.transform(result_icp_c.transformation)


    eva_1=o3d.pipelines.registration.evaluate_registration(u_temp,orim, 0.2)

    eva_2=o3d.pipelines.registration.evaluate_registration(oriu,orim, 0.2)


    print('mu',np.mean(dismu))
    print('um',np.mean(disum))

    print('after:',eva_1.inlier_rmse)
    print('after:',eva_1.fitness)

    print('before:',eva_2.inlier_rmse)
    print('before:',eva_2.fitness)


    trans=np.array([[0.956952631474, -0.191312983632, -0.218268156052, -0.219466581941],
    [0.177000641823, 0.980659842491, -0.083529040217, -0.364354223013],
    [0.230027005076, 0.041299730539, 0.972307503223, -0.477991551161],
    [0.000000000000, 0.000000000000, 0.000000000000 ,1.000000000000]])

    trans= inver_trans(trans)

    t_co=np.dot(trans,result_icp_c.transformation)
    t_=np.dot(trans,result_icp.transformation)

    ground_truth_matrix = np.eye(4)
    
    rotation_error, translation_error = calculate_errors(ground_truth_matrix, t_co)
    print(f"Rotation Error: {rotation_error} radians")
    print(f"Translation Error: {translation_error} units")


    rotation_err, translation_err = calculate_errors(ground_truth_matrix, t_)
    print(f"icpRotation Error: {rotation_err} radians")
    print(f"Translation Error: {translation_err} units")
    # pcd_orig1=density(pcd_orig)

    # draw([pcd_orig1.translate([-50,0,0]), pcd_orig])
    
    # pcd_orig1=uniformity(pcd_orig)

    # draw([pcd_orig1.translate([-50,0,0]), pcd_orig])



######################################
