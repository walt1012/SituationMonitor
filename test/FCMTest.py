import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt

# 数据集
n_samples = 3000
X = np.concatenate((
    np.random.normal((-1, -1, -1), size=(n_samples, 3)),
    np.random.normal((1, 1, 1), size=(n_samples, 3))
))

# fuzzy-c-means训练
fcm = FCM(n_clusters=2)
fcm.fit(X)

# 获取聚类中心和标签
fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)


# 画图函数
def Show(data, k, cp, cluster):
    num, dim = data.shape
    color = ['r', 'g', 'b']
    mark = ['x', 'o']
    ##三维图
    if dim == 3:
        ax = plt.subplot(111, projection='3d')
        for i in range(num):
            ax.scatter(data[i, 0], data[i, 1], data[i, 2], c=color[cluster[i]], marker=mark[cluster[i]])
    plt.savefig('images/basic-clustering-output.jpg')
    plt.show()


Show(X, 2, fcm_centers, fcm_labels)
