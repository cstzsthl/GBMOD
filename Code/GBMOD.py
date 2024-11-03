import numpy as np
import time
import warnings
import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import check_consistent_length, column_or_1d

import GB

warnings.filterwarnings('ignore')

class MS:
    def __init__(self,data, trandata, index, k):
             self.data=data
             self.trandata = trandata
             self.index = index
             self.k=k
             self.iteration_number = 3


    def getDst(self,data, data_shifted):
        j = 0
        dislist = []
        while (j <  len(data)):
            dst = np.sqrt(np.sum((data[j] - data_shifted[j]) ** 2))
            dislist.append(dst)
            j += 1
        return np.array(dislist)

    def k_nearest_neighbor(self,point, nbrs,points):
        distances, indices = nbrs.kneighbors([point])
        k_smallest = []
        for i,p in enumerate(indices[0]):
            k_smallest.append(points[p])
        del k_smallest[0]
        return k_smallest

    def get_medoid(self,coords):
        coords=np.array(coords)
        cost = distance.cdist(coords, coords, 'cityblock')# Manhattan distance,  'euclidean','minkowski'
        return coords[np.argmin(cost.sum(axis=0))]

    def get_mean(self,coords):
        return np.mean(coords, axis=0)

    def shift_iterationMean(self,points, k, iteration_number):
        shift_points = np.array(points)
        shift_points_COPY = shift_points[:]
        while(iteration_number>0):       # ['auto', 'brute','kd_tree', 'ball_tree']
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(shift_points_COPY)
            for i,point in enumerate(shift_points_COPY):
                KNearestNeighbor = self.k_nearest_neighbor(shift_points[i], nbrs,shift_points_COPY)
                shift_points[i] =self.get_mean(KNearestNeighbor)
                iteration_number -= 1
            shift_points_COPY = shift_points[:]
        return shift_points

    def shift_iterationMedoid(self,points, k, iteration_number):
        shift_points = np.array(points)
        shift_points_COPY = shift_points[:]
        while (iteration_number > 0):  # ['auto', 'brute','kd_tree', 'ball_tree']
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(shift_points_COPY)
            for i, point in enumerate(shift_points_COPY):
                KNearestNeighbor = self.k_nearest_neighbor(shift_points[i], nbrs, shift_points_COPY)
                shift_points[i] = self.get_medoid(KNearestNeighbor)  # mean or medoid
                iteration_number -= 1
            shift_points_COPY = shift_points[:]
        return shift_points

    def GBMOD(self):
        iteration = 1
        result = self.data[:]
        origin_results = np.zeros_like(self.trandata)
        while (iteration <= self.iteration_number):
            result = self.shift_iterationMean(result, self.k, 1)
            iteration += 1

        for i in range(0, len(self.index)):
            for j in self.index[i]:
                origin_results[int(j), :] = result[i, :]

        return self.getDst(self.trandata, origin_results)



data_path = "./Example.mat"
trandata = io.loadmat(data_path)['data']

oridata = trandata.copy()
trandata = trandata.astype(float)

scaler = MinMaxScaler()
trandata[:] = scaler.fit_transform(trandata[:])

X = trandata[:]

centers, gb_list, gb_weight = GB.getGranularBall(X)
index = []
for gb in gb_list:
    index.append(gb[:, -1])

k = 3
detector = MS(centers, X, index, k)
out_scores = detector.GBMOD()

out_scores = column_or_1d(out_scores)
for score in out_scores:
    print(score)
