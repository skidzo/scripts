# -*- coding: utf-8 -*-
"""
2012

calculation of Karhunen- Loève Transformation aka. PCA
"""

import numpy as np
from scipy.signal import detrend

def kl_transform(a):
    """
    Returns Karhunen Loeve Transform of the input and the transformation matrix and eigenval

    Ex:
    import numpy as np
    a  = np.array([[1,2,4],[2,3,10]])

    kk,m = KLT(a)
    print kk
    print m

    # to check, the following should return the original a
    print np.dot(kk.T,m).T

    """
    val,vec = np.linalg.eig(np.cov(a))
    klt = np.dot(vec,a)
    return klt,vec,val


class Klt:
    def __init__(self, dim, SnapshotSize):
        self.dim = dim
        self.kl_num_ts = SnapshotSize
        self.tau = 0

    def transform_fit(self, data, dhead=0):
        """
        KL-decomposition returning the eigenvectors and eigenvalues
        ordered by descending value,
        e.g.: the greatest Eigenvalue of snapshot i is eigw[i,0]
        The Karhunen-Loève Transformation in some special cases is comparable to the fast fourier Transformation,
        The main difference is that KLT uses signal-independent trancendent functions whereas the fft always is an approximation by signal dependent functions.

        :param dhead: number of colums in the beginning to exclude from analysis
        :param
        """
        n=0
        num_is = int(np.floor((len(data[:,1]) - 1) / self.kl_num_ts))
        #initialization of covariance matrix
        m = np.mat(np.zeros((self.dim, self.dim)))
        #initialization of meanvalue matrix
        m_data = np.mat(np.zeros((num_is, self.dim)))
        #initialization of Eigenvalue matrix
        eigw = np.array(np.zeros((num_is, self.dim)))
        #initialization of Modal Vektor Blocks: eigv(Snapshotindex, Componentindex, Eigenvalueindex)
        eigv = np.array(np.zeros((num_is, self.dim, self.dim)))
        #main calculation loop
        for j in range(0, num_is):
            # Calculate the mean acceleration values
            #a.mean(axis=0) # the mean of each of the 4 columns
            m_data[j, 0:self.dim] = np.mat(np.mean(data[n + 1 : n + self.kl_num_ts, dhead : dhead+self.dim], axis=0))
            # Initializing the summation matrice
            sigma = np.mat(np.zeros((self.dim, self.dim)))
            # Presum the acceleration value matrice
            for l in range(n+1, n+self.kl_num_ts):
                sigma = sigma + np.dot(np.mat(data[l, dhead : dhead + self.dim]).H, np.mat(data[l, dhead : dhead + self.dim]))
            n += self.kl_num_ts
            # Compute the covariance matrice
            m = (sigma / self.kl_num_ts) - np.dot(m_data[j, 0:self.dim].H , m_data[j, 0:self.dim])
            # Compute the Eigenvectors V and Eigenvalues D so that M*V = V*D
            d, v = np.linalg.eigh(m[0:self.dim, 0:self.dim])
            # order the eigenvalues!
            ind = np.argsort(d)[::-1]
            #store the Eigenvalues in a plottable matrice
            eigw[j, 0:self.dim] = d[ind]
            eigv[j, 0:self.dim, 0:self.dim] = v[:, ind]
            if j==num_is - 1:
                pass
        return eigw, eigv, m_data

    def transform_o(self, data,
                    dhead=0,
                    noverlapp=float(2./3.)):
        """
        KL-decomposition with overlapping returning the eigenvectors and eigenvalues
        ordered by descending value,
        e.g.: the greates Eigenvalue of snapshot i is eigw[i,0]

        :param dhead: number of colums in the beginning to exclude from analysis
        :param noverlapp: overlapping used by kl decomposition bringing it further to the quasi-continuus approach

        """
        n = 0
        num_is = int(np.floor((len(data[:,1]) - 1) / np.floor(self.kl_num_ts * (1. - noverlapp))))
        #print num_is
        #initialization of covariance matrix
        m = np.mat(np.zeros((self.dim, self.dim)))
        #initialization of meanvalue matrix
        m_data = np.mat(np.zeros((num_is, self.dim)))
        #initialization of Eigenvalue matrix
        eigw = np.array(np.zeros((num_is, self.dim)))
        #initialization of Modal Vektor Blocks: eigv(Snapshotindex, Componentindex, Eigenvalueindex)
        eigv = np.array(np.zeros((num_is, self.dim, self.dim)))
        #main calculation loop
        for j in range(0, num_is - 2):
            # Calculate the mean acceleration values
            #a.mean(axis=0) # the mean of each of the 4 columns
            m_data[j, 0:self.dim] = np.mat(np.mean(data[n + 1 : n + self.kl_num_ts, dhead : dhead + self.dim], axis=0))
            # Initializing the summation matrice
            sigma = np.mat(np.zeros((self.dim, self.dim)))
            # Presum the acceleration value matrice
            for l in range(n+1, n+self.kl_num_ts):
                sigma = sigma + np.dot(np.mat(data[l, dhead : dhead + self.dim]).H , np.mat(data[l, dhead : dhead + self.dim]))
            n += int(np.floor(self.kl_num_ts * (1.-noverlapp)))
            # Compute the covariance matrice
            m = (sigma / self.kl_num_ts) - np.dot(m_data[j, 0 : self.dim].H , m_data[j, 0 : self.dim])
            # Compute the Eigenvectors V and Eigenvalues D so that M*V = V*D
            d, v = np.linalg.eigh(m[0 : self.dim, 0 : self.dim])
            # order the eigenvalues!
            ind = np.argsort(d)[::-1]
            #store the Eigenvalues in a plottable matrice
            eigw[j, 0:self.dim] = d[ind]
            eigv[j, 0:self.dim, 0:self.dim] = v[:, ind]

            if j==num_is-2:
                pass
        return eigw, eigv, m_data

    def transform_tau(self, data,
                      dhead=0,
                      noverlapp=float(2./3.),
                      tau=None):
        """
        KL-decomposition with overlapping returning the eigenvectors and eigenvalues
        ordered by descending value,
        e.g.: the greates Eigenvalue of snapshot i is eigw[i,0]

        :param dhead: number of colums in the beginning to exclude from analysis
        :param noverlapp: overlapping used by kl decomposition bringing it further to the quasi-continuus approach

        """
        if not tau == None:
            self.tau = tau

        print("Testing Tau-Trafo", self.kl_num_ts,"Schrittweite, Tau:",self.tau)
        n=0


        num_is = int(np.floor((len(data[:,1])-1)/np.floor(self.kl_num_ts*(1.-noverlapp))))
        #initialization of covariance matrix
        m = np.mat(np.zeros((self.dim, self.dim)))
        #initialization of meanvalue matrix
        m_data = np.mat(np.zeros((num_is, self.dim)))
        #initialization of Eigenvalue matrix
        eigw = np.array(np.zeros((num_is, self.dim)))
        #initialization of Modal Vektor Blocks: eigv(Snapshotindex, Componentindex, Eigenvalueindex)
        eigv = np.array(np.zeros((num_is, self.dim, self.dim)))
        #main calculation loop
        for j in range(0, num_is-3):
            # Calculate the mean acceleration values
            #a.mean(axis=0) # the mean of each of the 4 columns
            m_data[j,0:self.dim] = np.mat(np.mean(data[n+1:n+self.kl_num_ts, dhead:dhead+self.dim], axis=0))
            # Initializing the summation matrice
            sigma = np.mat(np.zeros((self.dim, self.dim)))
            # Presum the acceleration value matrice
            for l in range(n+1, n+self.kl_num_ts):
                sigma = sigma + np.dot(np.mat(data[l, dhead:dhead+self.dim]).H , np.mat(data[l+self.tau, dhead:dhead+self.dim]))
            n += int(np.floor(self.kl_num_ts*(1.-noverlapp)))
            # Compute the covariance matrice
            m = (sigma / self.kl_num_ts) - np.dot(m_data[j, 0:self.dim].H , m_data[j, 0:self.dim])
            #print "Matrix M"
            #print M
            # Compute the Eigenvectors V and Eigenvalues D so that M*V = V*D
            d, v = np.linalg.eigh(m[0:self.dim, 0:self.dim])
            # order the eigenvalues!
            ind = np.argsort(d)[::-1]
            #store the Eigenvalues in a plottable matrice
            eigw[j, 0:self.dim] = d[ind]
            eigv[j, 0:self.dim, 0:self.dim] = v[:, ind]

            if j==num_is-2:
                pass
        return eigw, eigv, m_data

    def alpha(self,eigv, m_data,indlow=0,indup=-1):
        """
        backtransformation to discrete weigthting factors alpha
        """
        eigv = eigv[indlow:indup,:,:]
        m_data = m_data[indlow:indup,:]
        num_is = len(m_data[:,0])
        alpha = np.array(np.zeros((num_is, self.dim, self.dim)))
        print("transforming the weighting factors...")
        for j in range(num_is):
            alpha[j,0:self.dim,0:self.dim] = np.dot(np.asmatrix(eigv[j,0:self.dim,0:self.dim]).H , np.asmatrix(m_data[j,0:self.dim]).H)
        return alpha

    def pom(self, data):
        """
        compute proper orthogonal modes
        """

        n_data = len(data[:,0])

        if self.kl_num_ts > n_data:
            return np.zeros((self.dim,)), np.zeros((self.dim,self.dim))

        x = np.asmatrix(detrend(data[:,-self.dim:],axis=0,type='linear',bp=self.kl_num_ts))
        #print('detrended process has shape:', X.shape, 'and size in memory:', humanize_bytes(X.nbytes))
        # scaled covariance of the parameter estimate X
        m = (1./n_data) * np.dot(x.H , x)
        # Compute the Eigenvectors V and Eigenvalues D so that M*V = V*D
        d, v = np.linalg.eigh(m);
        # order the eigenvalues!
        ind = np.argsort(d)[::-1]
        #store the Eigenvalues in a plottable matrice
        pom = d[ind]
        pov = v[:, ind]
        #print(M, pom, pov)
        return pom, pov
