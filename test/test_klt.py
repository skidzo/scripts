# test_klt.py
import sys

sys.path.append("../klt")

from unittest import TestCase
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from klt import Klt

def get_iris_data():
    df = "../iris.csv"
    return np.genfromtxt(df,
                         delimiter=',',
                         skip_header=True)[:,:-1]

class Test_Klt(TestCase):
    def test_iris(self):
        pass

def main():
    data = get_iris_data()
    eigw, eigv, m_acc = Klt(dim=4, SnapshotSize=4).transform_fit(data)

    pca = PCA(n_components=4)
    pc = pca.fit_transform(StandardScaler().fit_transform(data))

    print(pca.explained_variance_ratio_)

    print(pc[1])

    #print(eigw[1])
    print(eigv[1])

    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax1.plot(Eigv)
    #plt.show()

if __name__ == '__main__':
    main()
