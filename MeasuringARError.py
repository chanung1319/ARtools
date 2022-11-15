# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:43:08 2022

@author: Chan-Ung Park
"""


from ObjLoader import ObjLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


#%% 3D Calculus 함수.

# 점에서 평면에 내린 수선의 발 (origin에서는 체크 했음....추가 체크 필요함)
def foot_of_perpendicular_from_point_to_plane(normal_of_plane, point_on_plane, arrival_point):
    n = normal_of_plane
    n_x = n[0]
    n_y = n[1]
    n_z = n[2]
    r0 = point_on_plane
    p0 = arrival_point
    p0_x = p0[0]
    p0_y = p0[1]
    p0_z = p0[2]
    
    if n_x == 0:
        if n_y == 0:
            foot_x = p0_x
            foot_y = p0_y
            foot_z = r0[2]
            foot = [foot_x, foot_y, foot_z]
            return foot
        else:
            foot_x = p0_x
            foot_y = (n_y*(np.dot(n, r0)) + p0_y*n_y**2 - n_y*n_z*p0_z)/(np.dot(n, n))
            foot_z = (n_z/n_y)*(foot_y - p0_y) + p0_z
            foot = [foot_x, foot_y, foot_z]
            return foot
    else:
        foot_x = (n_x*(np.dot(n, r0)) + p0_x*(n_y**2 + n_z**2) - n_x*n_y*p0_y - n_x*n_z*p0_z)/(np.dot(n, n))
        foot_y = (n_y/n_x)*(foot_x - p0_x) + p0_y
        foot_z = (n_z/n_x)*(foot_x - p0_x) + p0_z
        foot = [foot_x, foot_y, foot_z]
        return foot

# test_point가 평면으로부터 normal방향쪽 위에 존재하는지 (True), 아닌지 (False)를 판별.
def is_point_exist_above_plane(normal_of_plane, point_on_plane, test_point):
    foot_on_plane = foot_of_perpendicular_from_point_to_plane(normal_of_plane, point_on_plane, test_point)
    if (test_point[0] - foot_on_plane[0])/normal_of_plane[0] > 0:
        # 여기서 if 문 안쪽을 x, y, z에 대해서 모두 같은 값이 안나오면 crash를 시키고 싶은데 방법이???
        # raise ValueError 사용......"http://daplus.net/python-%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%97%90%EC%84%9C-%EC%98%88%EC%99%B8%EB%A5%BC-%EC%88%98%EB%8F%99%EC%9C%BC%EB%A1%9C-%EB%B0%9C%EC%83%9D-throw/"
        return True
    else:
        return False

# 한 평면에서부터 법선 방향의 반대방향으로 있는 모든 점만 찾아 새로운 point cloud로 출력.
def points_beneath_plane(normal_of_plane, point_on_plane, input_3d_point_set):
    output_set = []
    for i in range(0, int(input_3d_point_set.size/3)):
        p = input_3d_point_set[i]
        if is_point_exist_above_plane(normal_of_plane, point_on_plane, p):
            continue
        else:
            output_set.append(p)
    return np.array(output_set, dtype = 'float32')

def get_points_in_facial_area(input_3d_point_cloud, inferior_boundary_point, superior_boundary_point, left_boundary_point, right_boundary_point):
    superior = superior_boundary_point - inferior_boundary_point
    right = right_boundary_point - left_boundary_point
    anterior = np.cross(superior, right)
    # superior 경계와 inferior 경계 사이의 점들만 남기고 제거
    points_between_sup_inf_boundary = points_beneath_plane(-superior, inferior_boundary_point, points_beneath_plane(superior, superior_boundary_point, input_3d_point_cloud))
    # right 경계와 left 경계 사이의 점들 남기고 제거.
    points_between_sup_inf_rt_lt = points_beneath_plane(-right, left_boundary_point, points_beneath_plane(right, right_boundary_point, points_between_sup_inf_boundary))
    # posterior쪽 경계보다 더 posterior에 있는 점들 제거.
    facial_area = points_beneath_plane(-anterior, (left_boundary_point + right_boundary_point)/2, points_between_sup_inf_rt_lt)
    return facial_area


#%% 3D Plot 관련 함수.

def plot_3d_array_ptcloud(output_pcl, color):
    xdata = []
    ydata = []
    zdata = []
    for i in range(0, int(output_pcl.size / 3)):
        xdata.append(output_pcl[i][0])
        ydata.append(output_pcl[i][1])
        zdata.append(output_pcl[i][2])
        
    fig = plt.figure(figsize=(12, 12)) #(15, 15) 하니까 좋긴 함.
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xdata, ydata, zdata, marker='o', s=0.5, c=color)
    plt.show()
    return

def plot_two_3d_array_ptcloud(pcl1, pcl2, color1, color2):
    x1data = []
    y1data = []
    z1data = []
    for i in range(0, int(pcl1.size / 3)):
        x1data.append(pcl1[i][0])
        y1data.append(pcl1[i][1])
        z1data.append(pcl1[i][2])
    
    x2data = []
    y2data = []
    z2data = []
    for i in range(0, int(pcl2.size / 3)):
        x2data.append(pcl2[i][0])
        y2data.append(pcl2[i][1])
        z2data.append(pcl2[i][2])
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1data, y1data, z1data, marker='o', s=0.5, c=color1)
    ax.scatter(x2data, y2data, z2data, marker='o', s=0.5, c=color2)
    
    plt.show()
    return


#%% Step 1. Obj 파일을 point clouod array로 load.
    
# Registration이 완료된 데이터들을 obj로 새로 저장하여 입력.
ct_file_path = "C:/Users/skia-pc/Desktop/obj_transformer/315_ct_section.obj" # 파일 경로 입력.
scene_file_path = "C:/Users/skia-pc/Desktop/obj_transformer/315_scene_section.obj"

# Anatomical Landmarks.
menton_pos = np.array([0.124113, -0.215454, -0.266042])
glabella_pos = np.array([0.243005, -0.195572, -0.192005])
lt_lobule_pos = np.array([0.142855, -0.263964, -0.166639])
rt_lobule_pos = np.array([0.225038, -0.274315, -0.281074])

ct_indices, ct_buffer, ct_v, ct_vt, ct_vn = ObjLoader.load_model_without_face(ct_file_path, sorted=False)
sc_indices, sc_buffer, sc_v, sc_vt, sc_vn = ObjLoader.load_model_without_face(scene_file_path, sorted=False)

del ct_file_path, scene_file_path
del ct_buffer, ct_vt, ct_indices, sc_buffer, sc_vt, sc_indices
del ct_vn, sc_vn

ct_point_array = ObjLoader.array_to_vector3_array(ct_v)
scene_point_array = ObjLoader.array_to_vector3_array(sc_v)

# Load한 Point Cloud Visualizing
plot_two_3d_array_ptcloud(ct_point_array, scene_point_array, 'blue', 'orange')


#%% Step 1-1. 얼굴 영역 추출하기. (자동 절단이 아직 만족스럽지 않아서, 숨김.)

'''
# CT skin model 에서 얼굴 영역 추출.
ct_face_point_cloud = get_points_in_facial_area(ct_point_array, menton_pos, glabella_pos, lt_lobule_pos, rt_lobule_pos)
# 3D camera model 에서 얼굴 영역 추출.
scene_face_point_cloud = get_points_in_facial_area(scene_point_array, menton_pos, glabella_pos, lt_lobule_pos, rt_lobule_pos)
'''

#%% Step 2. kNN만으로 closest-point-pair distance들의 평균값 빠르게 계산 (두 clouod 사이 대략적인 거리)

from KNNHelper import KNNHelper
(no_model_error_set, closest_point_idx, closest_point_set) = KNNHelper.get_cloud_to_cloud_kNN(ct_point_array, scene_point_array)

mean_dist = np.mean(no_model_error_set)
std_dist = np.std(no_model_error_set)

print('No-Modeling -> mean distance = ', round(mean_dist * 1000, 3), ' mm')
print('No-Modeling -> std deviation = ', round(std_dist * 1000, 3), ' mm')


#%%

# 평면과, 평면 밖의 점 사이의 거리.
# 평면에 대한 정보는 normal, 평면위의 한 point를 이용해 입력.
def dist_between_plane_and_point(normal_of_plane, point_on_plane, compared_point):
    foot_on_plane = foot_of_perpendicular_from_point_to_plane(normal_of_plane, point_on_plane, compared_point)
    foot_compared_point_diff = foot_on_plane - compared_point
    return (np.dot(foot_compared_point_diff, foot_compared_point_diff))**0.5

# Least Square Plane을 찾아준다. SVD를 이용해서 어떻게 찾는지도 좀 더 공부해 볼 것.
def least_square_plane_Fitting(point_cloud, point_on_plane):
    import numpy as np
    from numpy.linalg import svd
    points = np.reshape(point_cloud, (np.shape(point_cloud)[0], -1)) # Collapse trialing dimensions
    assert points.shape[1] <= points.shape[0], "There are only {} points in {} dimensions.".format(points.shape[0], points.shape[1])
    x = points.T - point_on_plane[:,np.newaxis]
    M = np.dot(x, x.T)
    normal_of_plane = svd(M)[0][:,-1]
    return point_on_plane, normal_of_plane

def point_to_least_square_plane(comparing_point, reference_cloud, kNN):
    from KNNHelper import KNNHelper
    pair_point = KNNHelper.get_fast_kNN_from_3D_point_to_cloud(comparing_point, reference_cloud)
    kNN_of_pair_point = KNNHelper.get_fast_kNN_in_single_cloud(pair_point, reference_cloud, kNN)
    
    (p0, normal_of_ref_pair) = least_square_plane_Fitting(kNN_of_pair_point, pair_point)
    error = dist_between_plane_and_point(normal_of_ref_pair, p0, comparing_point)
    return error

#%% Step 3. Compared Cloud의 모든 점에 대해 iteration을 돌리며 점 사이 거리를 출력.

def get_point2point_error_with_least_square_plane_modeling(reference_cloud, compared_cloud, kNN):
    num_of_point_of_compared = int(compared_cloud.size / 3)
    #print('Numbur of points in Compared Point Cloud = ', num_of_point_of_compared)
    error_set = []
    for i in range(0, num_of_point_of_compared):
        compared_point = compared_cloud[i]
        point2plane_dist = point_to_least_square_plane(compared_point, reference_cloud, kNN)
        print('Current Points : ', (i+1), ' / ', num_of_point_of_compared,', ... error = ', round(point2plane_dist * 1000, 2), ' mm')
        error_set.append(point2plane_dist)
    rms_error = (np.dot(np.array(error_set), np.array(error_set)) / num_of_point_of_compared)**0.5
    return error_set, rms_error


(error_set, rms_error) = get_point2point_error_with_least_square_plane_modeling(scene_point_array, ct_point_array, 6)
print('SUMMARY')
print('RMS Error = ', round(rms_error * 1000, 3), '  mm')
print('Mean Error = ', round(np.mean(error_set) * 1000, 3), '  mm')
print('Std of Error = ', round(np.std(error_set) * 1000, 3))
