## GBMOD
Shitong Cheng, Xinyu Su, Baiyang Chen, Hongmei Chen, Dezhong Peng, and **Zhong Yuan***

## Abstract
Outlier detection is a crucial data mining task involving identifying abnormal objects, errors, or emerging trends. Mean-shift-based outlier detection techniques evaluate the abnormality of an object by calculating the mean distance between the object and its $k$-nearest neighbors. However, in datasets with significant noise, the presence of noise in the $k$-nearest neighbors of some objects makes the model ineffective in detecting outliers. Additionally, the mean-shift outlier detection technique depends on finding the $k$-nearest neighbors of an object, which can be time-consuming. To address these issues, we propose a granular-ball computing-based mean-shift outlier detection method (GBMOD). Specifically, we first generate high-quality granular-balls to cover the data. By using the centers of granular-balls as anchors, the subsequent mean-shift process can effectively avoid the influence of noise points in the neighborhood. Then, outliers are detected based on the distance from the object to the displaced center of the granular-ball to which it belongs. Finally, the distance between the object and the shifted center of the granular-ball to which the object belongs is calculated, resulting in the outlier scores of objects. Subsequent experiments demonstrate the effectiveness, efficiency, and robustness of the method proposed in this paper.

## Usage
You can run GBMOD.py:
```
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
```

You can get outputs as follows:
```
anomaly_scores =
0.974 0.939 1.042 1.039 1.078 0.057 0.016 0.039 0.047 0.061
0.124 0.042 0.135 0.054 0.116 0.016 0.126 0.043 0.089 0.039
0.089 0.148 0.036 0.063 0.049 0.070 0.078 0.015 0.020 0.027
0.024 0.109 0.049 0.062 0.010 0.017 0.078 0.048 0.032 0.044
0.069 0.098 0.130 0.026 0.086 0.054 0.026 0.019 0.079 0.014
```

## Citation
If you find GBMOD useful in your research, please consider citing:
```
@article{cheng2024gbmod,
  title={GBMOD: A granular-ball mean-shift outlier detector},
  author={Cheng, Shitong and Su, Xinyu and Chen, Baiyang and Chen, Hongmei and Peng, Dezhong and Yuan, Zhong},
  journal={Pattern Recognition},
  pages={111115},
  year={2024},
  publisher={Elsevier}
}
```
## Contact
If you have any questions, please contact yuanzhong@scu.edu.cn.
