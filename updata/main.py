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



if __name__ == "__main__":
    # path_data = '../startfrom_nov_uncer/83bm.ply'
    # path_data = 'fil_u.ply'
    path_data = 'uav.ply'
    # path_data = 'mmob.ply'
    # path_data ='data/cone.ply'
    filter_type = "high" # or 'all'

    rate=0.1


    scale_min_dist = 3
    scale_max_dist = 10
    max_neighbor = 60
    var = None#None # float number or None
    method = 'topk' # 'topk' or 'prob'

    pcd_orig = read_data(path_data)
    pcd_orig.paint_uniform_color([0, 0, 0])

    n_samples = int(len(pcd_orig.points)*rate)
    print("sampled_point", n_samples)

    # pcd_orig1=density(pcd_orig)

    # draw([pcd_orig1.translate([-50,0,0]), pcd_orig])
    
    # pcd_orig1=uniformity(pcd_orig)

    # draw([pcd_orig1.translate([-50,0,0]), pcd_orig])

    pcd_sampled = sample_pcd(pcd_orig, 
                             filter_type, 
                             n_samples, 
                             scale_min_dist, 
                             scale_max_dist,
                             max_neighbor, 
                             var, 
                             method
                            )



    # # Split the point cloud into batches
    # batches = split_point_cloud(pcd_orig, batch_size=1000000) # Adjust the batch size as needed

    # # Process each batch
    # processed_batches = process_batches(batches)

    # # Optionally, combine the processed batches
    # pcd_sampled = combine_batches(processed_batches)


    

    o3d.io.write_point_cloud('sub_sampledu.ply',pcd_sampled)

    pcd_sampled.paint_uniform_color([1, 0, 0])

    draw([pcd_orig.translate([-50,0,0]), pcd_sampled])
    