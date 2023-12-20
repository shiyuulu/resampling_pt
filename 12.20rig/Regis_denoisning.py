import os
import math
import numpy as np
import argparse
import open3d as o3d
import MinkowskiEngine as ME
import torch
import typing as t
import util.transform_estimation as te

from urllib.request import urlretrieve
from model.resunet import ResUNetBN2C
from util.visualization import get_colored_point_cloud_feature
from lib.eval import find_nn_gpu, visualize_nearest_neighbors

import open3d as o3d
import numpy as np

import os
import datetime
import laspy
import copy
import re

from lap_batch import sample_pcd, paint_points, fine_reg, calculate_errors, merge_point_clouds


NN_MAX_N = 500
SUBSAMPLE_SIZE = 10000



if not os.path.isfile('ResUNetBN2C-16feat-3conv.pth'):
    print('Downloading weights...')
    urlretrieve(
        "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
        'ResUNetBN2C-16feat-3conv.pth')



###############################################3

def get_transformation_matrix(theta,axis,translation):
    # Create rotation matrix for random rotation and translation
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d 
    R = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    transformation = np.eye(4)  # 4x4 identity matrix
    transformation[:3, :3] = R
    transformation[:3, 3] = translation
   
    return transformation



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)

    def draw_regis(source, target):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([source_temp, target_temp]  )
        
    draw_regis(source_temp,target_temp)

    return source_temp, target_temp 

def display_inlier_outlier(cloud, ind):
    """
    draw the inlier and outlier of statistic outlier filter
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
                                      

def getpcd(las,offset=[0,0,0]):
    '''
    generate open3d point cloud from laspy
    '''
    las.points.offsets=offset
    points = np.vstack((las.x, las.y, las.z)).transpose()
    colors = np.vstack((las.red, las.green, las.blue)).transpose()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 65535)
    return pcd


#########################################33

def points_to_pointcloud(
        points: np.array, voxel_size: float = 0.025, scalars: t.Optional[np.array] = None
) -> o3d.geometry.PointCloud():
    """ 
    convert numpy array points to open3d.PointCloud
    :param points: np.ndarray of shape (N, 3) representing floating point coordinates
    :param voxel_size: float
    :param scalars: (optional) np.ndarray of shape (N, 1), scalar of each point (e.g. FDI)
    :return: open3d.PointCloud
    """
    radius_normal = voxel_size * 2
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pcd.estimate_covariances(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    if scalars is not None:
        colors = np.asarray([int_to_rgb(i) for i in scalars])
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def int_to_rgb(val: int, min_val: int = 11, max_val: int = 48, norm: bool = True):
    if val > max_val:
        raise ValueError("val must not be greater than max_val")
    if val < 0 or max_val < 0:
        raise ValueError("arguments may not be negative")
    if val < min_val:
        raise ValueError("val must be greater than min_val")

    i = (val - min_val) * 255 / (max_val - min_val)
    r = round(math.sin(0.024 * i + 0) * 127 + 128)
    g = round(math.sin(0.024 * i + 2) * 127 + 128)
    b = round(math.sin(0.024 * i + 4) * 127 + 128)
    if norm:
        r /= 255
        g /= 255
        b /= 255
    return [r, g, b]


def draw_tensor_points(
        tensors: t.List[t.Union[np.ndarray, torch.Tensor]],
        extract_points: bool = False,
        uniform_color: bool = True,
        color_arrays: t.Optional[t.List[t.Union[np.ndarray, list]]] = None,
        min_color: t.Optional[float] = 11,
        max_color: t.Optional[float] = 48,
) -> None:
    """
    :param tensors: list of tensor, either numpy or torch with each having a shape of (N, 3) or (N, 4)
    :param extract_points: bool, to extract the points (N, 0:3)
    :param uniform_color: bool, apply uniform color
    :param color_arrays: (optional) list of np.ndarray or list of shape (N, 1) that contains point color
    :param min_color: (optional) float of min color for colormap
    :param max_color: (optional) float of max color for colormap
    :return:
    """
    if not isinstance(tensors, list):
        tensors = [tensors]
    if not isinstance(color_arrays, list):
        color_arrays = [color_arrays]

    if len(color_arrays) < len(tensors):
        color_arrays += [None] * (len(tensors) - len(color_arrays))

    pcd_list = []
    for tt, ca in zip(tensors, color_arrays):
        if isinstance(tt, torch.Tensor):
            tt = tt.clone().numpy()
        elif isinstance(tt, np.ndarray):
            tt = tt.copy()
        else:
            raise ValueError(
                "Tensor type not supported, should be torch.Tensor or np.ndarray"
            )
        np_points = np.squeeze(tt)
        if extract_points:
            np_points = np_points[:, 0:3]
        pcd_temp = points_to_pointcloud(np_points)
        if uniform_color and ca is None:
            pcd_temp.paint_uniform_color(list(np.random.uniform(size=3)))
        elif ca is not None:
            if min_color is None:
                min_color = np.min(ca)
            if max_color is None:
                max_color = np.max(ca)
            colors = np.asarray(
                [int_to_rgb(i, min_val=min_color, max_val=max_color) for i in ca]
            )
            pcd_temp.colors = o3d.utility.Vector3dVector(colors)
        pcd_list.append(pcd_temp)
    o3d.visualization.draw_geometries(pcd_list)


def apply_transformation(
        points: t.Union[np.ndarray, torch.Tensor],
        transformation: t.Union[np.ndarray, torch.Tensor],
) -> t.Union[np.ndarray, torch.Tensor]:
    """
    :param points: tensor of shape (N, 3) representing floating point coordinates
    :param transformation: (4, 4) tensor of a transformation matrix
    :return: transformed points
    """
    if all(isinstance(i, np.ndarray) for i in [points, transformation]):
        transformed_points = np.matmul(
            transformation,
            np.concatenate(
                [points[:, 0:3], np.ones(shape=(points.shape[0], 1))], axis=-1
            ).T,
        ).T
    elif all(isinstance(i, torch.Tensor) for i in [points, transformation]):
        transformed_points = torch.matmul(
            transformation,
            torch.concat(
                [points[:, 0:3], torch.ones(size=(points.shape[0], 1))], dim=-1
            ).T,
        ).T
    else:
        raise TypeError("Both inputs should be either np.ndarray or torch.Tensor type.")
    points[:, 0:3] = transformed_points[:, 0:3]
    return points


def find_corr(xyz0, xyz1, F0, F1, subsample_size=-1):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=NN_MAX_N)

    
    if subsample_size > 0 and subsample:
       
        #visualize_nearest_neighbors(xyz0[inds0], xyz1,  nn_inds)
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        #visualize_nearest_neighbors(xyz0, xyz1,  nn_inds)
        return xyz0, xyz1[nn_inds]


def alignment(config):
    '''
    inference with FCGF to get registraction result
    input:config
    return:    
        alined source point cloud
        
        transformation matrix
        
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    voxel_size = config.voxel_size
    checkpoint = torch.load(config.model)

    # init model
    model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)


    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)

    input_pcd = o3d.io.read_point_cloud(config.input)
    input2_pcd = o3d.io.read_point_cloud(config.input2)

    # create target input (points and features)
    fixed_xyz = np.array(input_pcd.points)
    fixed_feats = np.ones((len(fixed_xyz), 1))

    # create source input (points and features)
    randu = np.array(input2_pcd.points)
    moving_feats = np.ones((len(randu), 1))

    # create target sparse tensor and model features
    # voxelize xyz and feats
    fixed_coords = np.floor(fixed_xyz / voxel_size)
    fixed_coords, fixed_inds = ME.utils.sparse_quantize(fixed_coords, return_index=True)
    # convert to batched coords compatible with ME
    fixed_coords = ME.utils.batched_coordinates([fixed_coords])
    fixed_unique_xyz = fixed_xyz[fixed_inds]
    fixed_feats = fixed_feats[fixed_inds]
    fixed_tensor = ME.SparseTensor(
        torch.tensor(fixed_feats, dtype=torch.float32),
        coordinates=torch.tensor(fixed_coords, dtype=torch.int32),
        device=device
    )

    # create moving sparse tensor and model features
    moving_coords = np.floor(randu / voxel_size)
    moving_coords, moving_inds = ME.utils.sparse_quantize(moving_coords, return_index=True)
    # convert to batched coords compatible with ME
    moving_coords = ME.utils.batched_coordinates([moving_coords])
    moving_unique_xyz = randu[moving_inds]
    moving_feats = moving_feats[moving_inds]
    moving_tensor = ME.SparseTensor(
        torch.tensor(moving_feats, dtype=torch.float32),
        coordinates=torch.tensor(moving_coords, dtype=torch.int32),
        device=device
    )

    # visualize inputs to be aligned
   # draw_tensor_points([fixed_unique_xyz, moving_unique_xyz])

    # get features of inputs | inference
    fixed_model_feats = model(fixed_tensor).F
    moving_model_feats = model(moving_tensor).F

    # compute correspondences and alignment
    xyz0_corr, xyz1_corr = find_corr(
        torch.tensor(moving_unique_xyz, dtype=torch.float32).to(device),
        torch.tensor(fixed_unique_xyz, dtype=torch.float32).to(device),
        moving_model_feats,
        fixed_model_feats,
        subsample_size=SUBSAMPLE_SIZE,
    )
    xyz0_corr, xyz1_corr = xyz0_corr.cpu(), xyz1_corr.cpu()

    # estimate transformation using the correspondences
    est_transformation = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

    # transform moving points
    aligned_xyz = apply_transformation(randu.copy(), est_transformation.numpy())

    # visualize the results
    draw_tensor_points([fixed_xyz, aligned_xyz])

    return aligned_xyz, fixed_xyz, est_transformation.numpy()
#



def makepa(ptgt,psrc,voxelsize):
    '''
    parser the argument 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        # default='redkitchen-20.ply',
        default=ptgt,
        type=str,
        help='path to a pointcloud file')
    parser.add_argument(
        '-i2',
        '--input2',
        default=psrc,
        type=str,
        help='path to a pointcloud file')
   # default='KITTIvisualizationconv1-5-nout16.pth',
    parser.add_argument(
        '-m',
        '--model',
        default='ResUNetBN2C-16feat-3conv.pth',
        #default='KITTIconv1-5-nout16.pth',
        type=str,
        help='path to latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=voxelsize,
        type=float,
        help='voxel size to preprocess point cloud')

    config = parser.parse_args()
    return config

def mea_dis(pcd1,pcd2):
    dis1=np.array(pcd1.compute_point_cloud_distance(pcd2))
    dis2=np.array(pcd2.compute_point_cloud_distance(pcd1))
    mean1=np.median(dis1)
    mean2=np.median(dis2)
    return (mean1+mean2)/2


def FCGF_regis(source, target,transformation_matrix, subfolder_path, GT):


    #statistic outlier removal
    # print("Statistical oulier removal")
    source_cl, ind = source.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=5.0)
    target_cl, ind2 = target.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=11.0)

    rand_source = copy.deepcopy(source_cl)
    rand_source.transform(transformation_matrix)

    source_path=os.path.join(subfolder_path,'source.ply')
    target_path=os.path.join(subfolder_path,'target.ply')

    o3d.io.write_point_cloud(source_path,rand_source)
    o3d.io.write_point_cloud(target_path,target_cl)

    config=makepa(target_path,source_path,0.2)

    _,_,trans_fea=alignment(config)

    # o3d.visualization.draw_geometries([rand_source, target_cl]  )

    # o3d.visualization.draw_geometries([source_cl, target_cl]  )

    combined_matrix= np.dot(trans_fea, transformation_matrix)

    r_error, t_error =calculate_errors(GT, combined_matrix)

    # draw_registration_result(source_cl,target_cl, GT)

    # draw_registration_result(source_cl,target_cl, combined_matrix)

    return trans_fea, r_error, t_error


# def simplify(source,target):

#     regis(lazu,lazm,conu,conm,stable_ob,subfolder_path,transformation_matrix,GT):
#     u_stabel_ind=list(np.where(np.in1d(conu,stable_ob))[0])
#     m_stabel_ind=list(np.where(np.in1d(conm,stable_ob))[0])

#     u_stable=lazu[u_stabel_ind]
#     m_stable=lazm[m_stabel_ind]

#     selected_mm=getpcd(m_stable)
#     selected_uav=getpcd(u_stable)


#     dis_t=target.compute_nearest_neighbor_distance()

#     dis_s=source.compute_nearest_neighbor_distance()

#     voxel_size=max(np.median(dis_t),np.median(dis_s))*0.5

#     print(voxel_size)

#     max_neighbor = 60
#     var = None#None # float number or None
#     method = 'topk' # 'topk' or 'prob'
#     filter_type = "high" # or 'all'

#     target_simplfy= sample_pcd(target, 
#                 filter_type, 
#                 voxel_size,                             
#                 max_neighbor, 
#                 var, 
#                 method
#                 )

#     o3d.visualization.draw_geometries([target, target_simplfy])
#     source_simplfy= sample_pcd(source, 
#                 filter_type, 
#                 voxel_size,                                     
#                 max_neighbor, 
#                 var, 
#                 method
#                 )
#     o3d.visualization.draw_geometries([source, source_simplfy])
#     return  source_simplfy,target_simplfy



def simplify(las_source,las_target,conu,conm,stable_ob,  radius=0.3, gamma=0.15  ):

    source_stabel_ind=list(np.where(np.in1d(conu,stable_ob))[0])
    target_stabel_ind=list(np.where(np.in1d(conm,stable_ob))[0])


   
    s_stable=las_source[source_stabel_ind]
    t_stable=las_target[target_stabel_ind]


    target=getpcd(t_stable)
    source=getpcd(s_stable)

    
    dis_t=target.compute_nearest_neighbor_distance()

    dis_s=source.compute_nearest_neighbor_distance()

  

    voxel_size=max(max(np.median(dis_t),np.median(dis_s))*0.6,0.04)

    target.voxel_down_sample(0.6*voxel_size)

    source.voxel_down_sample(0.6*voxel_size)

    print(voxel_size)


    max_neighbor = 60
    var = None#None # float number or None
    method = 'topk' # 'topk' or 'prob'
    filter_type = "high" # or 'all'

    target_simplfy= sample_pcd(target, 
                filter_type, 
                voxel_size,
                radius, 
                gamma,                             
                max_neighbor, 
                var, 
                method
                )
    
    target_vis=copy.deepcopy(target_simplfy)
    target_vis.paint_uniform_color([1,0,0])
    # o3d.visualization.draw_geometries([target, target_vis])
    source_simplfy= sample_pcd(source, 
                filter_type, 
                voxel_size, 
                radius, 
                gamma,                                       
                max_neighbor, 
                var, 
                method
                )
    source_vis=copy.deepcopy(target_simplfy)
    source_vis.paint_uniform_color([1,0,0])
    # o3d.visualization.draw_geometries([source, source_vis])

    # source_simplfy=merge_point_clouds(source_simplfy)
    # target_simplfy=merge_point_clouds(target_simplfy)
    # o3d.visualization.draw_geometries([source_simplfy])
    # o3d.visualization.draw_geometries([target_simplfy])


    # o3d.visualization.draw_geometries([source])
    # o3d.visualization.draw_geometries([target])

    return  source_simplfy,target_simplfy, source,target, voxel_size


def write_pcd(pcd_1,pcd_2,folder,f1_name,f2_name):
    path1=os.path.join(folder,f1_name)
    path2=os.path.join(folder,f2_name)
    o3d.io.write_point_cloud(path1,pcd_1)
    o3d.io.write_point_cloud(path2,pcd_2)


def evaluate(source,target,T):
    source.transform(T)


def uniformity(pcd):
    points = np.asarray(pcd.points)

  # Compute the nearest neighbor distance for each point using the built-in function
    distances = pcd.compute_nearest_neighbor_distance()
    distances_array = np.asarray(distances)

    # Calculate the variance of the distances
    variance = np.var(distances_array)
    mean=np.mean(distances_array)
    return variance,mean


def regis(selected_uav,selected_mm,uav_simplify,mm_simplify,subfolder_path,transformation_matrix,GT):
   
    trans=[]
    # write_pcd(selected_mm,selected_uav,subfolder_path,'sub_mm.ply','sub_uav.ply')

    print("without simplify")   
    stable_FCGF,  stable_r_error, stable_t_error=FCGF_regis(selected_uav,selected_mm,transformation_matrix,subfolder_path,GT)
    trans.append(stable_FCGF)

    # eva=o3d.pipelines.registration.evaluate_registration(selected_mm,selected_uav.transform(np.dot( stable_FCGF,transformation_matrix)), 0.2)

    print("rotation_error",stable_r_error)
    print("translation_error",stable_t_error)
    eva=o3d.pipelines.registration.evaluate_registration(selected_mm,selected_uav.transform(np.dot( stable_FCGF,transformation_matrix)), 0.2)
    print("inlier_RMSE",eva.inlier_rmse)
    print("inlier_ratio",eva.fitness)
    
    # write_pcd(selected_mm,selected_uav,subfolder_path,'sub_mm_simple.ply','sub_uav_simple.ply')
    
    print("simplify")
    stable_sim_FCGF, stable_sim_r_error, stable_sim_t_error=FCGF_regis(uav_simplify,mm_simplify,transformation_matrix,subfolder_path,GT)
    trans.append(stable_sim_FCGF)
    eva_simplify=o3d.pipelines.registration.evaluate_registration(mm_simplify,uav_simplify.transform(np.dot(stable_sim_FCGF,transformation_matrix)), 0.2)
    print("rotation_error",stable_sim_r_error)
    print("translation_error",stable_sim_t_error)
    print("inlier_RMSE",eva_simplify.inlier_rmse)
    print("inlier_ratio",eva_simplify.fitness)
    
    write_pcd(uav_simplify.transform(np.dot(stable_sim_FCGF,transformation_matrix)),selected_uav.transform(np.dot( stable_FCGF,transformation_matrix)),subfolder_path,'simplified_rig.ply',"nosimplify_rig.ply")

    return stable_sim_FCGF

def Simplify_selected_class(lazu,lazm,conu,conm,ob_list1,ob_list2, subfolder_path,radius, gamma ):
    uav_simplify_b, mm_simplify_b, uav_b, mm_b, voxel_size=simplify(lazu,lazm,conu,conm,ob_list1,radius, gamma )

    uav_simplify_s, mm_simplify_s, uav_s, mm_s,_=simplify(lazu,lazm,conu,conm,ob_list2,radius, gamma*1.1 )

    list_ob=ob_list1.extend(ob_list2)
    # To find the rest (unstable or not stable points)
    s_rest_indices = list(np.where(~np.in1d(conu, list_ob))[0])
    t_rest_indices = list(np.where(~np.in1d(conm, list_ob))[0])
    
    s_rest = lazu[s_rest_indices]
    t_rest = lazm[t_rest_indices]
    t_rest=getpcd(t_rest)
    s_rest=getpcd(s_rest)
   

    uav_merge=merge_point_clouds(uav_b,uav_s,s_rest.voxel_down_sample(voxel_size))
    uav_merge_simplify=merge_point_clouds(uav_simplify_b,uav_simplify_s,s_rest.voxel_down_sample(voxel_size))
    mm_merge=merge_point_clouds(mm_b,mm_s,t_rest.voxel_down_sample(voxel_size))
    mm_merge_simplify=merge_point_clouds(mm_simplify_b,mm_simplify_s,t_rest.voxel_down_sample(voxel_size))

    #  source=merge_point_clouds(source,)
    # target=merge_point_clouds(target,t_rest.voxel_down_sample(voxel_size))
    trans=np.eye(4)
    draw_registration_result(uav_merge_simplify,uav_merge,trans)
    # o3d.visualization.draw_geometries([uav_merge_simplify, mm_merge_simplify]  )

    draw_registration_result(mm_merge_simplify,mm_merge,trans)

    write_pcd(uav_merge_simplify,mm_merge_simplify, subfolder_path,'uav_simplify.ply',"mm_simplify.ply")

    return uav_merge, mm_merge, uav_merge_simplify, mm_merge_simplify


path="/home/lu/Test/Thesis/Data/new/LAS"

obj={
   '0':'Unknown',
   '1':'Building',
   '2':'Wall',
   '3':'Conifer',
   '4':'Deciduous_tree',
   '5':'Shrubs',
   '6':'Hedge',
   '7':'Grass',
   '8':'Other_vegetation',
   '9':'Sealed_surface',
   '10':'Cobbles',
   '11':'Open_Surface',
   '12':'Car',
   '13':'Dynamic Object',
}

m_8212 = np.array([
    [0.999996542931, 0.001256729825, -0.002305272268, -0.373379528522],
    [-0.001255050651, 0.999998927116, 0.000729724357, 0.597876548767],
    [0.002306187060, -0.000726828584, 0.999997079372, -0.987902462482],
    [0.0, 0.0, 0.0, 1.0]
])

m_8237 = np.array([
    [0.999999940395, 0.000001023820, -0.000271826633, 0.062832102180],
    [-0.000001238840, 0.999999701977, -0.000791022030, 0.217279776931],
    [0.000271825731, 0.000791022321, 0.999999642372, -1.088942527771],
    [0.0, 0.0, 0.0, 1.0]
])

m_8258 = np.array([
    [0.999999940395, 0.000488824677, 0.000384396466, -0.025148568675],
    [-0.000488786551, 0.999999940395, -0.000099506899, 0.019945736974],
    [-0.000384444982, 0.000099319543, 0.999999940395, -0.180003792048],
    [0.0, 0.0, 0.0, 1.0]
])

m_8222 = np.array([
    [0.999992787838, 0.003489476629, -0.001559515134, -0.8631439209],
    [-0.003493126249, 0.999991178513, -0.002343644854, 0.8359202519],
    [0.001551322988, 0.002349075861, 0.999996006489, -1.250249862671],
    [0.0, 0.0, 0.0, 1.0]
])

m_8306 = np.array([
    [0.999999582767, 0.000496857625, 0.000866086746, -0.487767606974],
    [-0.000497441913, 0.999999821186, 0.000674791110, -0.033678267151],
    [-0.000865751179, -0.000675221439, 0.999999463558, 0.484542757273],
    [0.0, 0.0, 0.0, 1.0]
])

GT_matrix = {
    '8212': m_8212,
    '8237': m_8237,
    '8258': m_8258,
    '8222': m_8222,
    '8306': m_8306
}


def main():
    #root path
    for file_name in os.listdir(path):

        if file_name.endswith('.las'):
            
            file_path = os.path.join(path, file_name)

            #prepare file path          
            match = re.search(r'\d{4}', file_name)
            if match:
                number = match.group(0)
                print(number)
                GT=GT_matrix[number]
                save_folder = os.path.join(path, number)
                os.makedirs(save_folder, exist_ok=True)
            else:
                print(f"No matching number format found in {file_name}. Skipping...")
                continue

            #set seeds for random rotation 
            seed = np.random.randint(0, 2**32 - 1)  # 32-bit integer
            np.random.seed(seed)
            
            # angle = np.random.uniform(0, 1/2*np.pi)  # Random angle between 0 and 2*pi
            theta = np.random.uniform(0, 0.5*np.pi)  # Random rotation angle
            axis = np.random.normal(0, 1, 3)  # Random rotation axis
            translation = np.random.uniform(-100, 100, 3)  # Random translation between -10 and 10 for x, y, z
            
            transformation_matrix = get_transformation_matrix(theta, axis, translation)

            timestamp = "simplify_rig"
            subfolder_path = os.path.join(save_folder, timestamp)
            os.makedirs(subfolder_path, exist_ok=True)


            # Save seeds to a file for reproducibility
            with open(os.path.join(subfolder_path, 'seeds.txt'), 'w') as fs:
                fs.write(str(seed) + '\n')


            #read las point cloud
            laz = laspy.read(file_path)
            lazu=laz[laz.points['Original cloud index']==0]

            lazm=laz[laz.points['Original cloud index']==1]
            offset=lazm.points.offsets
            orim=getpcd(lazm)
            oriu=getpcd(lazu)
            trans=[]

            write_pcd(oriu,orim,subfolder_path,'uav.ply','mm.ply')

            print("whole point cloud")
            ori_FCGF, r_error, t_error=FCGF_regis(oriu,orim, transformation_matrix, subfolder_path,GT)
            trans.append(ori_FCGF)
            print("rotation_error",r_error)
            print("translation_error",t_error)
            eva=o3d.pipelines.registration.evaluate_registration(oriu,orim.transform(np.dot( ori_FCGF,transformation_matrix)), 0.2)
            print("inlier_RMSE",eva.inlier_rmse)
            print("inlier_ratio",eva.fitness)
            

       
             #select the stable target and apply P2Plane ICP
            conu=np.asarray(lazu.points['Constant'])
            conm=np.asarray(lazm.points['Constant'])

            stable_ob=[0,1,2]
            stable_os=[9,10,11]


            print('Building & road')

            uav_sub, mm_sub, uav_sub_simplify, mm_sub_simplify= Simplify_selected_class(lazu,lazm,conu,conm,stable_ob,stable_os, subfolder_path,0.5,0.12)

           
            trans_sub=regis(uav_sub, mm_sub, uav_sub_simplify, mm_sub_simplify, subfolder_path,transformation_matrix,GT)

            trans_ini=np.dot(trans_sub,transformation_matrix)

            fine_reg(uav_sub, mm_sub, oriu, orim, trans_ini, GT)

            # trans.append(trans_sub)
            # np.savetxt(os.path.join(subfolder_path, 'trans.txt'), trans,  delimiter=' ')
            



        


if __name__ == '__main__':
   
    main()
    #demo_alignment(config)