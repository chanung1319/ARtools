# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:16:30 2022

@author: Chan-Ung Park
"""

import numpy as np
from scipy.spatial import KDTree

class KNNHelper:
    @staticmethod
    def get_cloud_to_cloud_kNN(compared_cloud, reference_cloud, kNN = 1):
        kdtree = KDTree(reference_cloud)
        dist_of_knn, idxs_of_knn = kdtree.query(compared_cloud, k=kNN)
        closest_point_set = reference_cloud[idxs_of_knn]
        return dist_of_knn, idxs_of_knn, closest_point_set

    @staticmethod
    def get_fast_kNN_from_3D_point_to_cloud(single_point, point_cloud, kNN = 1):
        #참고 : https://stackoverflow.com/questions/54114728/finding-nearest-neighbor-for-python-numpy-ndarray-in-3d-space
        kdtree = KDTree(point_cloud)
        dist_of_knn, idxs_of_knn = kdtree.query(single_point, k=kNN)
    
        if kNN == 1:
            return point_cloud[idxs_of_knn]
        else:
            position_of_knn = np.zeros(shape=(kNN, 3))
            for i in range(0, idxs_of_knn.size):
                position_of_knn[i] = point_cloud[idxs_of_knn[i]]
            return position_of_knn

    @staticmethod
    def get_fast_kNN_in_single_cloud(center_point, point_cloud, kNN = 6):
        kdtree = KDTree(point_cloud)
        dist_of_knn, idxs_of_knn = kdtree.query(center_point, k=(kNN + 1))
    
        position_of_knn = np.zeros(shape=(kNN, 3))
        for i in range(1, idxs_of_knn.size):
            position_of_knn[i-1] = point_cloud[idxs_of_knn[i]]
        return position_of_knn
