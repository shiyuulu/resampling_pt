import open3d as o3d
from utils import read_data, draw, read_data
from graph_filter import sample_pcd, paint_points
import numpy as np
from tqdm import tqdm

'''


def density(pcd):
    
    point = np.asarray(pcd.points)           # 获取点坐标
    kdtree = o3d.geometry.KDTreeFlann(pcd)   # 建立KD树索引
    point_size = point.shape[0]              # 获取点的个数
    dd = np.zeros(point_size)
    for i in tqdm(range(point_size)):
        [_, idx, dis] = kdtree.search_knn_vector_3d(point[i],30)
        dd[i] = (dis[1])                      
   
    mean_density = np.median((dd))
    var_density = np.var((dd))

     # Select points with density larger than mean + 2 * var
    selected_indices = np.where(dd> mean_density )[0]

    print("Selected points count:", len(selected_indices)/point_size)



    print("点云密度为 denstity=", mean_density)

    pcd_selected=pcd.select_by_index(selected_indices)
    return  pcd_selected



def uniformity(pcd):
    points = np.asarray(pcd.points)
 
#     voxel_size=0.1
#     voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
#   # Adjust voxel_size as needed
#     voxel_counts = np.array([len(voxel.points) for voxel in voxel_grid.voxels])
# Assess the distribution of voxel_counts for uniformity

  # Compute the nearest neighbor distance for each point using the built-in function
    distances = pcd.compute_nearest_neighbor_distance()
    distances_array = np.asarray(distances)

    # Calculate the variance of the distances
    variance = np.var( distances_array)
    mean=np.median( distances_array)

    print("mean points count:", mean)



    print("var", variance)
    return variance,mean
'''

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

def process_batches(batches):
    """
    Processes each batch of the point cloud.

    Args:
    - batches: list of open3d.geometry.PointCloud

    Returns:
    - processed_batches: list of processed batches (e.g., sampled point clouds)
    """
    processed_batches = []
    for batch in batches:
        # Call your processing functions here
        # For example: sampled_batch = sample_pcd(batch, ...)

        pcd_sampled = sample_pcd(batch, 
                                filter_type, 
                                n_samples, 
                                scale_min_dist, 
                                scale_max_dist,
                                max_neighbor, 
                                var, 
                                method
                                )

        processed_batches.append(pcd_sampled)
    return processed_batches

def combine_batches(processed_batches):
    """
    Combines processed batches back into a single point cloud.

    Args:
    - processed_batches: list of processed point cloud batches

    Returns:
    - combined_pcd: open3d.geometry.PointCloud
    """
    combined_points = []
    for batch in processed_batches:
        combined_points.extend(np.asarray(batch.points))
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    return combined_pcd

def select_main(pcd, threshold=0.1):
    # Extract colors from the point cloud
    colors = np.asarray(pcd.colors)

    # Select points where the green channel is less than the specified threshold
    selected_indices = np.where(colors[:, 1] < 0.6   )[0]

    # Filter the point cloud based on the selected indices
    filtered_pcd = pcd.select_by_index(selected_indices)

    return filtered_pcd


def segment_planes_and_measure_thickness(point_cloud,number_of_surfaces_to_segment, distance_threshold=0.01, num_iterations=1000, ransac_n=3):
    all_planes = []
    segmented_pcds = []
    distances_list = []

    # Make a copy of the point cloud to work with
    pcd_temp = copy.deepcopy(point_cloud)

    # Segment out planes iteratively
    for i in range(number_of_surfaces_to_segment): # Set your number of expected surfaces here
        plane_model, inliers = pcd_temp.segment_plane(distance_threshold=distance_threshold,
                                                      ransac_n=ransac_n,
                                                      num_iterations=num_iterations)
        [a, b, c, d] = plane_model
        all_planes.append(plane_model)
        
        # Extract inliers as the segmented plane
        inlier_cloud = pcd_temp.select_by_index(inliers)
        segmented_pcds.append(inlier_cloud)
        
        # Calculate distance from inliers to plane (thickness)
        inlier_points = np.asarray(inlier_cloud.points)
        distances = np.abs((inlier_points @ np.array([a, b, c])) + d) / np.linalg.norm([a, b, c])
        distances_list.append(distances)   
        # Remove the inliers from the cloud
        pcd_temp = pcd_temp.select_by_index(inliers, invert=True)
        # Optionally visualize each segmented plane
        # o3d.visualization.draw_geometries([inlier_cloud])
    # Visualization of final segmented point cloud
    o3d.visualization.draw_geometries(segmented_pcds)
    
    return all_planes, distances_list, pcd_temp


import open3d as o3d
import numpy as np
import pyransac3d as pyrsc

def ransac_plane(pcd):
    # ------------------------------------参数设置---------------------------------------
    segment = []    # 存储分割结果的容器
    min_num = 20000  # 每个分割直线所需的最小点数
    dist = 0.1  # Ransac分割的距离阈值
    iters = 0       # 用于统计迭代次数，非待设置参数
    # -----------------------------------分割多个平面-------------------------------------
    while len(pcd.points) > min_num:

        points = np.asarray(pcd.points)
        plano1 = pyrsc.Plane()
        best_eq, inliers = plano1.fit(points, thresh=dist, maxIteration=100)

        plane_cloud = pcd.select_by_index(inliers)       # 分割出的平面点云
        r_color = np.random.uniform(0, 1, (1, 3))         # 平面点云随机赋色
        plane_cloud.paint_uniform_color([r_color[:, 0], r_color[:, 1], r_color[:, 2]])
        pcd = pcd.select_by_index(inliers, invert=True)  # 剩余的点云
        segment.append(plane_cloud)
        #file_name = "RansacFitMutiPlane" + str(iters + 1) + ".pcd"
        #o3d.io.write_point_cloud(file_name, plane_cloud)
        iters += 1
        if len(inliers) < min_num:
            break
    # ------------------------------------结果可视化--------------------------------------
    o3d.visualization.draw_geometries(segment, window_name="Ransac分割多个平面",
                                    width=1024, height=768,
                                    left=50, top=50,
                                    mesh_show_back_face=False)
    return segment




def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0.1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      
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

    return adjusted_colors


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

if __name__ == "__main__":
    # path_data = '../startfrom_nov_uncer/83bm.ply'
    # path_data = 'fil_u.ply'
    # path_data = 'uav.ply'
    path_data = 'm_curv_density.ply'
    path2='u_curv_density.ply'
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

    # u_match=match_histograms(u_color,m_color)
    m_match=match_histograms(m_color,u_color)
    # u_match= np.asarray(equalize_color_histogram(u_color))



    
    # Plotting the histograms
    plot_histograms(m_color, m_match, "Source Histogram Matching")
    # plot_histograms(u_match, m_color, "Target Histogram")
    m_temp = copy.deepcopy(pcd_orig)

    m_temp.colors=o3d.utility.Vector3dVector(m_match)
    o3d.visualization.draw_geometries([m_temp])
    o3d.visualization.draw_geometries([pcd_orig])

    n_samples = int(len(pcd_orig.points)*rate)
    print("sampled_point", n_samples)

    orim=o3d.io.read_point_cloud("m.ply")
    oriu=o3d.io.read_point_cloud("u.ply")

    dismu=orim.compute_point_cloud_distance(oriu)

    disum=oriu.compute_point_cloud_distance(orim)
    # o3d.visualization.draw_geometries([pcd_u]) 

    m_main=select_main(m_temp, threshold=0.1)
    u_main=select_main(pcd_u, threshold=0.1)
    
    o3d.visualization.draw_geometries([m_main])  # Uncomment to visualize

    o3d.visualization.draw_geometries([u_main])  # Uncomment to visualize


    
    # # Define the number of surfaces you expect to segment
    # number_of_surfaces_to_segment = 8 # Example value, adjust to your scenario

    # # Run the function
    # planes, thicknesses,pcd = segment_planes_and_measure_thickness(m_main, number_of_surfaces_to_segment=number_of_surfaces_to_segment)

    # planes, thicknesses,pcd2 = segment_planes_and_measure_thickness(u_main, number_of_surfaces_to_segment=number_of_surfaces_to_segment)

    # pcd=ransac_plane(m_main)
    # pcd2=ransac_plane(u_main)
    # draw initial alignment
    current_transformation = np.identity(4)

    downu = u_main.voxel_down_sample(voxel_size=0.01)
    downm = m_main.voxel_down_sample(voxel_size=0.01)


    downu.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))
    downm.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=60))

    # result_icp = o3d.pipelines.registration.registration_icp(
    # downu, downm, 0.1, current_transformation,
    # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    # )
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        downu, downm, 0.06, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=50))
    print(result_icp)
    draw_registration_result_original_color(oriu, orim,
                                            result_icp.transformation)

    u_temp=copy.deepcopy(oriu)

    u_temp.transform(result_icp.transformation)


    eva_1=o3d.pipelines.registration.evaluate_registration(u_temp,orim, 0.2)

    eva_2=o3d.pipelines.registration.evaluate_registration(oriu,orim, 0.2)


    print('mu',np.mean(dismu))
    print('um',np.mean(disum))

    print('after:',eva_1.inlier_rmse)
    print('after:',eva_1.fitness)

    print('before:',eva_2.inlier_rmse)
    print('before:',eva_2.fitness)


    # pcd_orig1=density(pcd_orig)

    # draw([pcd_orig1.translate([-50,0,0]), pcd_orig])
    
    # pcd_orig1=uniformity(pcd_orig)

    # draw([pcd_orig1.translate([-50,0,0]), pcd_orig])





    # pcd_sampled = sample_pcd(pcd_orig, 
    #                          filter_type, 
    #                          n_samples, 
    #                          scale_min_dist, 
    #                          scale_max_dist,
    #                          max_neighbor, 
    #                          var, 
    #                          method
    #                         )



    # # # Split the point cloud into batches
    # # batches = split_point_cloud(pcd_orig, batch_size=1000000) # Adjust the batch size as needed

    # # # Process each batch
    # # processed_batches = process_batches(batches)

    # # # Optionally, combine the processed batches
    # # pcd_sampled = combine_batches(processed_batches)


    

    # o3d.io.write_point_cloud('sub_sampledm.ply',pcd_sampled)

    # pcd_sampled.paint_uniform_color([1, 0, 0])

    # draw([pcd_orig.translate([-50,0,0]), pcd_sampled])
    