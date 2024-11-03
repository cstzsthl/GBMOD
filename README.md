## GBMOD
Shitong Cheng, Xinyu Su, Baiyang Chen, Hongmei Chen, Dezhong Peng, and **Zhong Yuan***

## Abstract
Outlier detection is a crucial data mining task involving identifying abnormal objects, errors, or emerging trends. Mean-shift-based outlier detection techniques evaluate the abnormality of an object by calculating the mean distance between the object and its $k$-nearest neighbors. However, in datasets with significant noise, the presence of noise in the $k$-nearest neighbors of some objects makes the model ineffective in detecting outliers. Additionally, the mean-shift outlier detection technique depends on finding the $k$-nearest neighbors of an object, which can be time-consuming. To address these issues, we propose a granular-ball computing-based mean-shift outlier detection method (GBMOD). Specifically, we first generate high-quality granular-balls to cover the data. By using the centers of granular-balls as anchors, the subsequent mean-shift process can effectively avoid the influence of noise points in the neighborhood. Then, outliers are detected based on the distance from the object to the displaced center of the granular-ball to which it belongs. Finally, the distance between the object and the shifted center of the granular-ball to which the object belongs is calculated, resulting in the outlier scores of objects. Subsequent experiments demonstrate the effectiveness, efficiency, and robustness of the method proposed in this paper.

## Usage
You can run GBMOD.py:

You can get outputs as follows:
```
anomaly_scores =
    1.3717
    1.2816
    1.4044
    1.3226
    1.3673
    1.1867
    1.5351
    1.4129
    1.1125
    1.5381
```

## Citation
If you find GBMOD useful in your research, please consider citing:
```
@article{
}
```
## Contact
If you have any questions, please contact yuanzhong@scu.edu.cn.
