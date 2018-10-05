# test_klt.py
import sys

sys.path.append("../klt")

from unittest import TestCase
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from klt import Klt, kl_transform

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

    data = np.array([[1,2,4],[2,3,10]])

    #eigw, eigv, m_acc = Klt(dim=2, SnapshotSize=2).transform_fit(data)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(data)

    print(pca.explained_variance_ratio_)

    print(pc)

    #print(eigw[1])
    #print(eigv[1])

    #StandardScaler().fit_transform(data)

    klt,vec,val = kl_transform(data)

    print(val)
    print(vec)

    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax1.plot(Eigv)
    #plt.show()

if __name__ == '__main__':
    main()
