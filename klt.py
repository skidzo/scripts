# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:24:19 2012

FahrKomA

Berechnung der Karhunen- Loève Transformation

@author: eckstjo
"""

import numpy as np
from scipy.signal import detrend

class Klt:
    def __init__(self,dim,SnapshotSize):
        self.dim = dim
        self.kl_num_ts = SnapshotSize
        self.tau = 0
    
    def Transform(self,acc,dhead=0):
        """
        KL-decomposition returning the eigenvectors and eigenvalues 
        ordered by descending value,
        e.g.: the greatest Eigenvalue of snapshot i is Eigw[i,0]
        Die Karhunen-Loève Transformation ist ähnlich der Fourier Transformation, 
        der entscheidende Unterschied liegt bei erstgenannter in der Signalabhängigkeit der charakteristischen Funktionen, 
        wohingegen zweitgenannte signalunabhängige transzendente Funktionen verwendet, siehe \cite{Mertins96}.       
        
        """
        n=0
        num_is = int(np.floor((len(acc[:,1])-1)/self.kl_num_ts))
        #initialization of covariance matrix
        M = np.mat(np.zeros((self.dim, self.dim)))
        #initialization of meanvalue matrix
        m_acc = np.mat(np.zeros((num_is, self.dim)))
        #initialization of Eigenvalue matrix
        Eigw = np.array(np.zeros((num_is, self.dim)))
        #initialization of Modal Vektor Blocks: Eigv(Snapshotindex, Componentindex, Eigenvalueindex)
        Eigv = np.array(np.zeros((num_is, self.dim, self.dim)))
        #print 'KLD number of steps:', str(num_is)
        #print 'Initialized numerical arrays and matrices'
        #main calculation loop
        for j in range(0, num_is):
            # Calculate the mean acceleration values
            #a.mean(axis=0) # the mean of each of the 4 columns
            m_acc[j,0:self.dim] = np.mat(np.mean(acc[n+1:n+self.kl_num_ts, dhead:dhead+self.dim], axis=0))
            # Initializing the summation matrice
            Sigma = np.mat(np.zeros((self.dim, self.dim)))
            # Presum the acceleration value matrice
            for l in range(n+1, n+self.kl_num_ts):
                Sigma = Sigma + np.dot(np.mat(acc[l, dhead:dhead+self.dim]).H , np.mat(acc[l, dhead:dhead+self.dim]))
            n += self.kl_num_ts
            # Compute the covariance matrice
            M = (Sigma / self.kl_num_ts) - np.dot(m_acc[j, 0:self.dim].H , m_acc[j, 0:self.dim])
            #print "Matrix M"
            #print M
            # Compute the Eigenvectors V and Eigenvalues D so that M*V = V*D
            D,V = np.linalg.eigh(M[0:self.dim,0:self.dim])
            # order the eigenvalues!
            ind = (np.argsort(D))
            ind = ind[::-1]
            #store the Eigenvalues in a plottable matrice
            Eigw[j, 0:self.dim] = D[ind]
            Eigv[j, 0:self.dim, 0:self.dim] = V[:, ind]
            if j==num_is-1:
                pass
                #print num_is
                #print 'Done, last Core Iteration, iteration step:',str(j+1),'. '
                #print Eigw.shape, Eigv.shape
        return Eigw, Eigv, m_acc

    def OTransform(self,acc,dhead=0,noverlapp=float(2./3.)):
        """
        KL-decomposition with overlapping returning the eigenvectors and eigenvalues 
        ordered by descending value,
        e.g.: the greates Eigenvalue of snapshot i is Eigw[i,0]
        Die Karhunen-Loève Transformation ist in gewisser Weise ähnlich der Fourier Transformation, 
        der entscheidende Unterschied liegt bei erstgenannter in der Signalabhängigkeit der charakteristischen Funktionen, 
        wohingegen zweitgenannte signalunabhängige transzendente Funktionen verwendet, siehe \cite{Mertins96}.       
        
        :param dhead: number of colums in the beginning to exclude from analysis
        :param noverlapp: overlapping used by kl decomposition bringing it further to the quasi-continuus approach
        
        """
        n=0
        #print "without overlapp", self.kl_num_ts
        #print noverlapp
        #print "with overlapp", self.kl_num_ts*(1.-noverlapp)
        #print len(acc[:,1])
        
        num_is = int(np.floor((len(acc[:,1])-1)/np.floor(self.kl_num_ts*(1.-noverlapp))))
        #print num_is
        #initialization of covariance matrix
        M = np.mat(np.zeros((self.dim, self.dim)))
        #initialization of meanvalue matrix
        m_acc = np.mat(np.zeros((num_is, self.dim)))
        #initialization of Eigenvalue matrix
        Eigw = np.array(np.zeros((num_is, self.dim)))
        #initialization of Modal Vektor Blocks: Eigv(Snapshotindex, Componentindex, Eigenvalueindex)
        Eigv = np.array(np.zeros((num_is, self.dim, self.dim)))
        #print 'KLD number of steps:', str(num_is)
        #print 'Initialized numerical arrays and matrices'
        #main calculation loop
        for j in range(0, num_is-2):
            # Calculate the mean acceleration values
            #a.mean(axis=0) # the mean of each of the 4 columns
            m_acc[j,0:self.dim] = np.mat(np.mean(acc[n+1:n+self.kl_num_ts, dhead:dhead+self.dim], axis=0))
            # Initializing the summation matrice
            Sigma = np.mat(np.zeros((self.dim, self.dim)))
            # Presum the acceleration value matrice
            for l in range(n+1, n+self.kl_num_ts):
                Sigma = Sigma + np.dot(np.mat(acc[l, dhead:dhead+self.dim]).H , np.mat(acc[l, dhead:dhead+self.dim]))
            n += int(np.floor(self.kl_num_ts*(1.-noverlapp)))
            # Compute the covariance matrice
            M = (Sigma / self.kl_num_ts) - np.dot(m_acc[j, 0:self.dim].H , m_acc[j, 0:self.dim])
            #print "Matrix M"
            #print M
            # Compute the Eigenvectors V and Eigenvalues D so that M*V = V*D
            D,V = np.linalg.eigh(M[0:self.dim,0:self.dim])
            # order the eigenvalues!
            ind = (np.argsort(D))
            ind = ind[::-1]
            #store the Eigenvalues in a plottable matrice
            Eigw[j, 0:self.dim] = D[ind]
            Eigv[j, 0:self.dim, 0:self.dim] = V[:, ind]
            
            #print j
            if j==num_is-2:
                pass
                #print num_is
                #print 'Done, last Core Iteration, iteration step:',str(j+1),'. '
                #print Eigw.shape, Eigv.shape
        return Eigw, Eigv, m_acc

    def TauTransform(self,acc,dhead=0,noverlapp=float(2./3.),tau=None):
        """
        KL-decomposition with overlapping returning the eigenvectors and eigenvalues 
        ordered by descending value,
        e.g.: the greates Eigenvalue of snapshot i is Eigw[i,0]
        Die Karhunen-Loève Transformation ist in gewisser Weise ähnlich der Fourier Transformation, 
        der entscheidende Unterschied liegt bei erstgenannter in der Signalabhängigkeit der charakteristischen Funktionen, 
        wohingegen zweitgenannte signalunabhängige transzendente Funktionen verwendet, siehe \cite{Mertins96}.       
        
        :param dhead: number of colums in the beginning to exclude from analysis
        :param noverlapp: overlapping used by kl decomposition bringing it further to the quasi-continuus approach
        
        """
        if not tau == None:
            self.tau = tau
        
        print "Testing Tau-Trafo", self.kl_num_ts,"Schrittweite, Tau:",self.tau
        n=0
        #print "without overlapp", self.kl_num_ts
        #print noverlapp
        #print "with overlapp", self.kl_num_ts*(1.-noverlapp)
        #print len(acc[:,1])
        
        num_is = int(np.floor((len(acc[:,1])-1)/np.floor(self.kl_num_ts*(1.-noverlapp))))
        #print num_is
        #initialization of covariance matrix
        M = np.mat(np.zeros((self.dim, self.dim)))
        #initialization of meanvalue matrix
        m_acc = np.mat(np.zeros((num_is, self.dim)))
        #initialization of Eigenvalue matrix
        Eigw = np.array(np.zeros((num_is, self.dim)))
        #initialization of Modal Vektor Blocks: Eigv(Snapshotindex, Componentindex, Eigenvalueindex)
        Eigv = np.array(np.zeros((num_is, self.dim, self.dim)))
        #print 'KLD number of steps:', str(num_is)
        #print 'Initialized numerical arrays and matrices'
        #main calculation loop
        for j in range(0, num_is-3):
            # Calculate the mean acceleration values
            #a.mean(axis=0) # the mean of each of the 4 columns
            m_acc[j,0:self.dim] = np.mat(np.mean(acc[n+1:n+self.kl_num_ts, dhead:dhead+self.dim], axis=0))
            # Initializing the summation matrice
            Sigma = np.mat(np.zeros((self.dim, self.dim)))
            # Presum the acceleration value matrice
            for l in range(n+1, n+self.kl_num_ts):
                Sigma = Sigma + np.dot(np.mat(acc[l, dhead:dhead+self.dim]).H , np.mat(acc[l+self.tau, dhead:dhead+self.dim]))
            n += int(np.floor(self.kl_num_ts*(1.-noverlapp)))
            # Compute the covariance matrice
            M = (Sigma / self.kl_num_ts) - np.dot(m_acc[j, 0:self.dim].H , m_acc[j, 0:self.dim])
            #print "Matrix M"
            #print M
            # Compute the Eigenvectors V and Eigenvalues D so that M*V = V*D
            D,V = np.linalg.eigh(M[0:self.dim,0:self.dim])
            # order the eigenvalues!
            ind = (np.argsort(D))
            ind = ind[::-1]
            #store the Eigenvalues in a plottable matrice
            Eigw[j, 0:self.dim] = D[ind]
            Eigv[j, 0:self.dim, 0:self.dim] = V[:, ind]
            
            #print j
            if j==num_is-2:
                pass
                #print num_is
                #print 'Done, last Core Iteration, iteration step:',str(j+1),'. '
                #print Eigw.shape, Eigv.shape
        return Eigw, Eigv, m_acc
    
    def Alpha(self,Eigv, m_acc,indlow=0,indup=-1):
        """
        backtransformation to discrete weigthting factors alpha
        """
        Eigv = Eigv[indlow:indup,:,:]
        m_acc = m_acc[indlow:indup,:]
        num_is = len(m_acc[:,0])
        Alpha = np.array(np.zeros((num_is, self.dim, self.dim)));
        print "transforming the weighting factors..."
        #print "Alpha =", np.asmatrix(Eigv[0,0:4,0:4]).H, "*", np.asmatrix(m_acc[0,0:4]).H
        for j in range(num_is):
            Alpha[j,0:self.dim,0:self.dim] = np.dot(np.asmatrix(Eigv[j,0:self.dim,0:self.dim]).H , np.asmatrix(m_acc[j,0:self.dim]).H)
        return Alpha

    def Pom(self,acc):
        """
        compute proper orthogonal modes
        """
        
        n_acc = len(acc[:,0])
        
        if self.kl_num_ts > n_acc:
            return np.zeros((self.dim,)), np.zeros((self.dim,self.dim))
        
        X = np.asmatrix(detrend(acc[:,-self.dim:],axis=0,type='linear',bp=self.kl_num_ts))
        #print 'detrended process has shape:', X.shape, 'and size in memory:', humanize_bytes(X.nbytes)
        #print acc, X
        # scaled covariance of the parameter estimate X
        M = (1./n_acc) * np.dot(X.H , X)
        # Compute the Eigenvectors V and Eigenvalues D so that M*V = V*D
        D,V = np.linalg.eigh(M);
        # order the eigenvalues!
        ind = np.argsort(D)
        ind = ind[::-1]
        #store the Eigenvalues in a plottable matrice
        POM = D[ind]
        POV = V[:,ind]
        #print M, POM, POV
        return POM, POV
