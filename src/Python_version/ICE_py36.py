#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 05:47:03 2018

@author: zg
"""

import numpy as np
#from scipy import io
import scipy.io 
#import pickle 
from sklearn.model_selection import StratifiedKFold
#import sklearn
from scipy.sparse import spdiags
from scipy.spatial import distance
#import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier
from sklearn import svm
#from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import tree
import copy
import numpy.matlib
from sklearn.exceptions import NotFittedError

#import FuzzyRwrBagging as frb

#from joblib import Parallel, delayed
#import multiprocessing


def RWR(A, nSteps, laziness, p0 = None):
    '''
    % the random walk algorithm. 
    % A is the input net matrix, with the diag to be 0. 
    % nSteps: how many steps to walk
    % laziness: the probablity to go back. 
    % p0: the initial probability. usually it is a zero matrix with the diag to
    %   be 1. 
    %
    % for example, A could be: 
    %  A = [0,2,2,0,0,0,0;...
    %      2,0,1,1,0,0,0;...
    %      2,1,0,0,1,0,0;...
    %      0,1,0,0,0,1,1;...
    %      0,0,1,0,0,0,0;...
    %      0,0,0,1,0,0,1;...
    %      0,0,0,1,0,1,0]
    % 
    % if nSteps is 1000 and laziness is 0.3, p0 is default, the result is:
    %    [0.449, 0.207, 0.220, 0.064, 0.154, 0.034, 0.034;...
    %     0.207, 0.425, 0.167, 0.132, 0.117, 0.071, 0.071;...
    %     0.220, 0.167, 0.463, 0.052, 0.324, 0.028, 0.028;...
    %     0.048, 0.099, 0.039, 0.431, 0.027, 0.232, 0.232;...
    %     0.038, 0.029, 0.081, 0.009, 0.356, 0.004, 0.004;...
    %     0.017, 0.035, 0.014, 0.154, 0.009, 0.425, 0.203;...
    %     0.017, 0.035, 0.014, 0.154, 0.009, 0.203, 0.425]
    %
    % Each column represents the propability for each node. each element in the
    %  column means the probability to go to that node.
    % This algorithm will converge. For example, for the above matrix, nSteps =
    %  100, 1000 or 10000, will give the same result. 
    '''
    n = len(A)
    if p0 == None:
        p0 = np.eye(n)
    '''
    % In the example above, spdiags(sum(A)'.^(-1), 0, n, n) will be 
    %     0.2500         0         0         0         0         0         0
    %          0    0.2500         0         0         0         0         0
    %          0         0    0.2500         0         0         0         0
    %          0         0         0    0.3333         0         0         0
    %          0         0         0         0    1.0000         0         0
    %          0         0         0         0         0    0.5000         0
    %          0         0         0         0         0         0    0.5000
    
    % W will be:
    %          0    0.5000    0.5000         0         0         0         0
    %     0.5000         0    0.2500    0.3333         0         0         0
    %     0.5000    0.2500         0         0    1.0000         0         0
    %          0    0.2500         0         0         0    0.5000    0.5000
    %          0         0    0.2500         0         0         0         0
    %          0         0         0    0.3333         0         0    0.5000
    %          0         0         0    0.3333         0    0.5000         0
    '''
    #W = A * spdiags(sum(A)'.^(-1), 0, n, n);
    #W = spdiags(np.power(sum(np.float64(A)) , -1).T  , 0, n, n).toarray()
    W = A.dot( spdiags(np.power(sum(np.float64(A)) , -1)[np.newaxis], \
                       0, n, n).toarray() )
    p = p0
    pl2norm = np.inf
    unchanged = 0
    for i in range(1, nSteps+1):
        if i % 100 == 0:
            print('      done rwr ' + str(i-1) )
            
        pnew = (1-laziness) * W.dot(p) + laziness * p0
        l2norm = max(np.sqrt(sum((pnew - p) ** 2) ) )
        p = pnew
        if l2norm < np.finfo(float).eps:
            break
        else:
            if l2norm == pl2norm:
                unchanged = unchanged +1
                if unchanged > 10:
                    break
            else:
                unchanged = 0
                pl2norm = l2norm
    return p

# test RWR()
'''
A = np.array([[0,2,2,0,0,0,0],\
	[2,0,1,1,0,0,0],\
	[2,1,0,0,1,0,0],\
	[0,1,0,0,0,1,1],\
	[0,0,1,0,0,0,0],\
	[0,0,0,1,0,0,1],\
	[0,0,0,1,0,1,0]])

nSteps = 1000
lazi = 0.3
RWR(A, nSteps, lazi, None)
'''

# test
#dst = distance.euclidean(A)
# corrent, the same as in Matlab

def f_sim_2_aRankNet(sim, k=3):
    '''
    % Convert the similarity matrix to a network graph where each node 
    %  has k edges to other nodes (aRank).    
    '''
    # delete the diagnal values.
    # sim = sim-diag(diag(sim) );
    np.fill_diagonal(sim, 0)
    
    # [~, I] = sort(sim-diag(diag(sim) ) );
    I = np.argsort(sim, kind='mergesort') + 1
    
    # [~, I2] = sort(I);
    I2 = (np.argsort(I, kind='mergesort').T + 1).T
    
    # for every column, just keep the top k edges.
    #aRankNet = (I2 >length(sim)-k);
    aRankNet = I2 > (len(sim) - k)
    
    # make it a diagonal matrix
    # aRankNet = max(aRankNet, aRankNet');
    aRankNet = np.logical_or(aRankNet, aRankNet.T)
    
    # remove the diagonal 1s.
    # aRankNet = aRankNet-diag(diag(aRankNet) );
    np.fill_diagonal(aRankNet, False)
    
    return aRankNet
    
# test
#sim = np.array([[0,    0.5566,    0.6448,    0.3289], \
#                [0.5566,         0,   -0.0842,   -0.0170], \
#                [0.6448,   -0.0842,         0,    0.8405], \
#                [0.3289,   -0.0170,    0.8405,         0]])
#
#f_sim_2_aRankNet(sim,1)
#f_sim_2_aRankNet(sim,2)
#f_sim_2_aRankNet(sim,3)
#
#array([[False,  True,  True, False],
#       [ True, False, False, False],
#       [ True, False, False,  True],
#       [False, False,  True, False]])
#        
#array([[False,  True,  True,  True],
#       [ True, False, False, False],
#       [ True, False, False,  True],
#       [ True, False,  True, False]])
#        
#array([[False,  True,  True,  True],
#       [ True, False, False,  True],
#       [ True, False, False,  True],
#       [ True,  True,  True, False]])
        


def f_find_centers_rwMat(rw_mat, k):
    '''
    % on the rw_mat matrix, find some nodes as the centroids for soft
    %  clustering. If we just random pickup some nodes as centroids, that is
    %  not good for fuzzy clusters. 
    % k is the number of centroids. 
    '''
    ixs = []
    # 1. find the most connected center node as the first centroid. 
    a = np.sum(rw_mat, axis=1) # axis=1 for rows; 0 for col
    # % most connected node.
    ix = np.argmax(a)
    ixs.append(ix)
    
    # % 2. iteratively find the rest nodes
    for i in range(1, k):
        tmp = rw_mat[:, ixs]
        b = np.sum(tmp, axis=1)
        b[ixs] = np.inf
        
        # % find the farthest node
        ix = np.argmin(b)
        ixs.append(ix)
    return ixs

# test
#tmp = f_find_centers_rwMat(rw_mat, 10)

def getCutoff(rw_mat, avgNeighborsSize):
    tmp = rw_mat.flatten('F')
    a = np.flip(np.sort(tmp), 0)
    
    len1 = len(rw_mat)
    #cutoffs = []
    
    all_neibs = int( avgNeighborsSize * len1 )
    print( all_neibs)
    ct = a[all_neibs]
    return ct

#test 

#>>> a = np.array([[1,2], [3,4]])
#>>> a.flatten()
#array([1, 2, 3, 4])
#>>> a.flatten('F')
#array([1, 3, 2, 4])
'''
a = np.array( range(0,100) )
b = np.matlib.repmat(a, 100, 1)
ct = getCutoff(b, 70)
'''


def f_len_of_each_ele(c1):
    #% Assume c1 is a 1-dimension cell array, and each element is a 1d double
    #%  array. This function counts the length of each double array.
    lens = np.zeros(len(c1))
    for i in range(0, len(c1)):
        lens[i] = len(c1[i])
    
    return lens


def f_eu_dist(X):
    '''
    calculate the euclidean distance between instances
    '''
    sim = np.zeros(( len(X), len(X) ))
    for i in range(0, len(X)):
        for j in range(i+1, len(X)):
            tmp = distance.euclidean(X[i], X[j])
            sim[i][j] = tmp
            sim[j][i] = tmp
    sim = -sim
    np.fill_diagonal(sim, 0)
    
    return sim


#test
#sim = f_eu_dist(X)


def f_eu_dist2(X1, X2):
    '''
    calculate the euclidean distance between instances from two datasets
    '''
    sim = np.zeros(( len(X1), len(X2) ))
    for i in range(0, len(X1) ):
        for j in range(0, len(X2) ):
            tmp = distance.euclidean(X1[i], X2[j])
            sim[i][j] = tmp
    sim = -sim
    
    return sim


#test
#sim = f_eu_dist2(X_tr, X_te)



def f_fuzzy_rwr_clusters(X, k=100, each_clus_sz=None):
    # X: data
    # k: number of clusters
    '''
    The return variable clus stores the instance indices for each cluster. 
     However, this data structure is not easy to find for a instance, which are
     the clusters it belongs to, thus we also need to convert clus to a 
     true-false matrix. 
    '''
    
    if each_clus_sz == None:
        # on average, how many clusters does one inst belongs to. 
        #overlap_factor = 2;
        # the estimated size of each cluster. default is half the number of
        #  instances.
        each_clus_sz=len(X)/3
    
    print('RWR-based fuzzy clustering starts...')
    print('  NO. clusters = '+str(k)+'; avg. cluster size = '+str(each_clus_sz) )
    
    # sim = squareform(pdist(X)); 
    # sim = -sim;
    sim = np.zeros((len(X), len(X) ) )
    for i in range(0, len(X)):
        for j in range(i+1, len(X)):
            tmp = distance.euclidean(X[i], X[j])
            sim[i][j] = tmp
            sim[j][i] = tmp
    sim = -sim
    print('  done calculating the Euclidean distance matrix')
    
    # ---------------------------------------------------------------
    aRank_k_neighbors = np.ceil(np.log10(len(sim)) )
    ori_graph = f_sim_2_aRankNet(sim, aRank_k_neighbors)
    print('  done calculating the A-rank KNN graph')
    
    # % -------- RWR --------
    nSteps = 1000
    lazi = 0.3
    rw = RWR(ori_graph, nSteps, lazi)
    
    # remove probability of returning start node
    np.fill_diagonal(rw, 0)
    rw_mat = rw
    print('  done RWR')
    # ---------------------------------------------------------------
    
    ixs_centers = f_find_centers_rwMat(rw_mat, k)
    ct = getCutoff(rw_mat, each_clus_sz)
    rw_net = rw_mat > ct
    
    # % set the diagnal to 1
    np.fill_diagonal(rw_net, True)
    
    clus = []
    for i in range(0, k):
        tmp = np.argwhere(rw_net[:, ixs_centers[i] ] ).flatten()
        clus.append(tmp)
    
    # ---------------------------------------------------------------
    # % sort the clusters
    lens = f_len_of_each_ele(clus)
    ix = np.argsort(lens)[::-1]
    
    clus_ordered = [clus[i] for i in ix]
    
    print('  center inst. index of each cluster: ')
    ixs_centers = np.array(ixs_centers)
    print(ixs_centers[ix])
    print('  size of each cluster: ')
    print(lens[ix])
    
    print('  done RWR clustering')
    return clus_ordered

#test
#clus = f_fuzzy_rwr_clusters(X, 100)
# pass
    

def f_clus_to_tfs(clus, n_inst):
    #% convert the cluster information from cell array to mat. But for each
    #%  instance, the rank of clusters information will be lost - you won't know
    #%  what is the top 1/2/3 cluster it belongs to. 
    #% 
    #% clus e.g:
    #% 1x5 cell
    #% 1x195 double  1x193 double  1x169 double  1x161 double 1x62 double
    #%
    #% tfs e.g:
    #% 295x5 double
    #%      1     0     0     0     0
    #%      1     1     1     1     0
    #%      1     1     1     0     0
    #%      1     1     0     0     0
    #%      1     1     1     1     0
    #% ...
    #%      1     1     1     1     1
    #%      1     0     0     0     0
    #%      1     1     1     0     0
    tfs = np.zeros((n_inst, len(clus)), dtype=bool)
    for i in range(0, len(clus)):
        tfs[clus[i], i] = True
    
    return tfs
    
# test
#tfs = f_clus_to_tfs(clus, len(X))
# pass

def f_tfs_2_instClus(tfs):
    '''
    convert the boolean table representation of clustering result to for each 
     instance, what clusters it belongs to. 
    '''
    inst_clus = []
    
    for i in range(0, len(tfs)):
        row = list( np.where(tfs[i, :] ) [0] )
        inst_clus.append(row)
    
    return inst_clus

# test
#inst_clus = f_tfs_2_instClus(tfs)



#def f_bg_svm_tr_te(X_tr, y_tr, X_te, y_te):
#    #bagging = BaggingClassifier(base_estimator = svm.LinearSVC(), \
#    bagging = BaggingClassifier(base_estimator = tree.DecisionTreeClassifier(), \
#        random_state=None, n_estimators = 100 )
#    bagging.fit(X_tr, y_tr)
#    
#    y_pred = bagging.predict_proba(X_te)
#    y_pred = y_pred[:, 1].flatten()
#    
#    auc = roc_auc_score(y_te.flatten(), y_pred)
#    
#    return [y_pred, auc]

# test 
'''
X_tr = X
y_tr = y
X_te = X
y_te = y
[y_pred, auc] = f_bg_svm_tr_te(X_tr, y_tr, X_te, y_te)
'''

#def f_bg_tr_te(X_tr, y_tr, X_te, y_te, BaseBagging):
#    '''
#    corresponds to f_weka_bg_svm_tr_te() in Matlab version
#    '''
#    #bagging = BaggingClassifier(base_estimator = svm.LinearSVC(), \
#    bagging = BaggingClassifier(BaseBagging, \
#        random_state=None, n_estimators = 100 )
#    bagging.fit(X_tr, y_tr)
#    
#    y_pred = bagging.predict_proba(X_te)
#    y_pred = y_pred[:, 1].flatten()
#    
#    auc = roc_auc_score(y_te.flatten(), y_pred)
#    
#    return [y_pred, auc]


def f_tr(X_tr, y_tr, model):
    model_inner = copy.deepcopy(model)
    model_inner.fit(X_tr, y_tr)
    
    return model_inner


def f_te(X_te, model):
    y_pred = model.predict_proba(X_te)
    y_pred = y_pred[:, 1].flatten()
    
    return y_pred

    
def f_tr_te(X_tr, y_tr, X_te, model):
    '''
    corresponds to f_weka_bg_svm_tr_te() in Matlab version
    '''
    #bagging = BaggingClassifier(base_estimator = svm.LinearSVC(), \
    #bagging = BaggingClassifier(BaseBagging, \
    #    random_state=None, n_estimators = 100 )
    model_inner = copy.deepcopy(model)
    model_inner.fit(X_tr, y_tr)
    
    y_pred = model_inner.predict_proba(X_te)
    y_pred = y_pred[:, 1].flatten()
    
    #auc = roc_auc_score(y_te.flatten(), y_pred)
    
    return y_pred


def f_k_fo(X, y, model, k_fold=10):
    '''
    corresponds to f_weka_bg_svm_arff_k_fo_3_parfor() in Matlab version
    
    '''
    
    y = y.flatten()
    y_pred = np.zeros(y.size)
    
    skf = StratifiedKFold(n_splits=k_fold, random_state=None, shuffle=True)
    skf.get_n_splits(X, y)
    
    for train_index, test_index in skf.split(X, y):
        #print("TRAIN: ", train_index, "  TEST: ", test_index)
        X_tr, X_te = X[train_index], X[test_index]
        #y_tr, y_te = y[train_index], y[test_index]
        y_tr = y[train_index]
        
        if np.unique(y_tr).size == 1:
            y_pred_fo = np.zeros( len(test_index) )
            #print len(X_te)
            #print len(test_index)
            #print y_pred_fo
            y_pred_fo.fill(np.unique(y_tr)[0] )
            #print y_pred_fo
            
        else:
            y_pred_fo = f_tr_te(X_tr, y_tr, X_te, model)

        y_pred[test_index] = y_pred_fo
    
    #auc = roc_auc_score(y.flatten(), y_pred)
    return y_pred


# test
#pa = '/Volumes/Macintosh_HD/Users/zg/bio/3_ensembF/3_scripts/2017_4_4/'
##X = scipy.io.loadmat(pa+'/data/data_all_pickle/30/data.mat')['X'] # 30:breast cancer
##y = scipy.io.loadmat(pa+'/data/data_all_pickle/30/data.mat')['y']
#X = scipy.io.loadmat(pa+'/data/data_all_pickle/11/data.mat')['X'] # 11:mesothelioma
#y = scipy.io.loadmat(pa+'/data/data_all_pickle/11/data.mat')['y']
#
#model = BaggingClassifier(base_estimator = tree.DecisionTreeClassifier(), \
#    random_state=None, n_estimators = 100 )
#y_pred = f_k_fo(X, y, model, k_fold=10)
#
#print roc_auc_score(y.flatten(), y_pred)
# the easy dataset mesothelioma get 1.0 CV result.
# breast cancer get 0.599
# all results are correct.



def f_quantileNorm(templete, target):
    '''
    Templete is the standard, change the target to the values in the templete.
     Target may have a very different range than the templete. 
     
    templete and target should be 1d n by 1 array.
    
    f_my_quantileNorm()
    '''
    ix_target = np.argsort(target, kind='mergesort')
    ix_templete = np.argsort(templete, kind='mergesort')
    
    target[ix_target] = templete[ix_templete]
    
    new = target
    return new

# test
#templete = X[:, 0]
#target = X[:, 1]
#new = f_quantileNorm(templete, target)

#def f_bg_k_fo_3(X, y, k_fold=10):
#    '''
#    corresponds to f_weka_bgSvm_arff_k_fo_3_parfor() in Matlab version
#    corresponds to f_k_fo()
#    '''
#    y_pred = np.zeros((y.size, 1))
#    
#    skf = StratifiedKFold(n_splits=k_fold)
#    skf.get_n_splits(X, y)
#    
#    for train_index, test_index in skf.split(X, y):
#        #print("TRAIN:", train_index, "TEST:", test_index)
#        X_tr, X_te = X[train_index], X[test_index]
#        y_tr, y_te = y[train_index], y[test_index]





def f_use_each_clus_forWhole(X, y, clus, y_pred_whole, model, fo_inner):
    '''
    % using each cluster data to predict the whole instances, while self 
    %  prediction using 10-fold CV.
    
    corresponds to f_use_each_clus_forWhole_bg_svm() in Matlab version
    
    '''
    
    n_clusters = len(clus)
    y_pred_multi = np.zeros((y.size, n_clusters) )
    models = []
    
    for j in range(0, n_clusters):
        # for each cluster
        Xj = X[clus[j].flatten(), :]
        yj = y[clus[j].flatten() ]
        model_a_clust = copy.deepcopy(model)
        
        print(' Cluster '+str(j)+' started...')
        #if len(yj) > 10:
        if len(yj) > 15  and  np.unique(yj).size != 1:
            # ------------------ for self ------------------
            #if np.unique(yj).size == 1:
            #    y_pred = np.zeros(yj.size)
            #    y_pred.fill(np.unique(yj)[0])
            #else:
            
            try:
                y_pred = f_k_fo(Xj, yj, model, fo_inner)
                
                
                # quantileNorm
                templete = y_pred_whole[clus[j].flatten()]
                target = y_pred
                y_pred = f_quantileNorm(templete, target)
                
                # copy the normed prediction to the whole data.
                y_pred_multi[clus[j].flatten(), j] = y_pred
                
                print('  c-'+str(j)+' done predicting local instances')
                
                # ------------------ for other -----------------
                ix_other = set(range(0, y.size)) - set(clus[j].flatten()) 
                ix_other = list(ix_other)
                #print ix_other
                X_other = X[ix_other , :]
                #y_other = y[ix_other ]
                # predict
                #y_pred = f_tr_te(Xj, yj, X_other, model)
                #if np.unique(yj).size != 1:   
                model_a_clust.fit(Xj, yj)
                y_pred = model_a_clust.predict_proba(X_other)
                y_pred = y_pred[:, 1].flatten()
                
                # quantileNorm
                templete = y_pred_whole[ix_other]
                target = y_pred
                y_pred = f_quantileNorm(templete, target)
                #else:
                #    y_pred = np.zeros(X_other.size)
                #    y_pred.fill(np.unique(yj)[0])  
                
                # copy to the whole array
                y_pred_multi[ix_other, j] = y_pred
                print('  c-'+str(j)+' done predicting remote instances')
            except ValueError as e:
                print(e)
                print('  skip this cluster')
                y_pred = np.zeros(y.size)
                y_pred.fill(np.nan)
                y_pred_multi[:, j] = y_pred
                
        else:
            if len(yj) <= 15:
                print ('  '+str(len(yj))+' insts in cluster, <= 15, skip...')
                y_pred = np.zeros(y.size)
                y_pred.fill(np.nan)
                y_pred_multi[:, j] = y_pred
            
            if np.unique(yj).size == 1:
                print ('  warning, #unique class label(s) == 1')
                y_pred = np.zeros(y.size)
                y_pred.fill(np.unique(yj)[0]) 
                y_pred_multi[:, j] = y_pred
                
                model_a_clust = np.unique(yj)[0]
                        
        models.append(model_a_clust)
        
    return [y_pred_multi, models]


# test
#[y_pred_multi, models] = f_use_each_clus_forWhole(X, y, clus, y_pred_whole, model)



#def f_dec_tab_4_bg_svm(X, y, clus):
#    '''
#    Calculate the decision table
#    % This version changed from the cluster-cluster dec_mat to instance-cluster
#    %  dec_mat. This solution will avoid the case that if one cluster decision 
#    %  is wrong leading entrie cluster prediction is wrong, which is the reason
#    %  of instability. However, we cannot use a systematic evaluation criteria 
#    %  such as AUC, I will try using the predicted prob at first. 
#    
#    % This version 3 adds the support for fuzzy clustering - one instance may
#    %  belongs to more than one cluster. 
#    % This updated version also outputs the predicted values of y.
#    % support more than 3 clusters
#    % normalization take place in y_pred_self and y_pred_other, thus do not
#    %  need normalization when predict y_pred_ICE.
#    % ixsp is another cluster form.
#    
#    corresponds to f_dec_tab_4_bg_svm() in Matlab version
#    '''
#    #n_clusters = len(clus)
#    ## dec_mat stores the prediction error.
#    #pred_mat=np.zeros((y.size, n_clusters+1)) #the extra col is for whole pred
#    # 
#    ## k_fold of inner cross-validation
#    #fo_inner = 10
#    # --------------------------- WHOLE -------------------------
#    
#    # --------------------------- SELF -------------------------



def f_err_mat(X, y, clus, model):
    '''
    Calculate the decision table
    
    corresponds to f_dec_tab_4_bg_svm() in Matlab version
    
    '''
    n_clusters = len(clus)
    # err_mat stores the prediction error.
    pred_prob_mat=np.zeros((y.size, n_clusters+1)) #the extra col is for whole pred
    # col 0 to col n_clusters-1 store the predictions by each cluster
    # the last col stores the pred by whole data
    
    #models = []
    
    # k_fold of inner cross-validation
    fo_inner = 5
    
    # --------------------------- WHOLE -------------------------
    # Predict each cluster using the whole data. 
    model_whole = copy.deepcopy(model)
    y_pred_whole = f_k_fo(X, y, model_whole, fo_inner)
    model_whole.fit(X, y) # fit a model using all data rather than only a fold
    
    pred_prob_mat[:, n_clusters] = y_pred_whole
    print (' Done evaluation using whole instances')
    print (' Start to evaluate each cluster ')
    
    # --------------------------- SELF -------------------------
    # predict the whole instances using each cluster data, while self 
    #  prediction using 10-fold CV.
    
    [y_pred_multi, models] = f_use_each_clus_forWhole(X, y, clus, \
        y_pred_whole, model, fo_inner)
    print (' Done evaluation using each cluster')
    models.append(model_whole)
    
    pred_prob_mat[:, 0:n_clusters] = y_pred_multi
    
    # make a tmp array a stores y
    tmp = np.matlib.repmat(y.reshape((y.size, 1)), 1, n_clusters+1)
    err_mat = abs(pred_prob_mat - tmp )
    
    print (' Done calculating error table and fitting ICE models')
    return [err_mat, models]


"""
#mat = scipy.io.loadmat('/Volumes/Macintosh_HD/Users/zg/bio/3_ensembF/'+\
#                      '3_scripts/2017_4_4/data/names.mat')['names']
#mat = io.loadmat('/Users/zg/Desktop/a.mat')['names']

#test
pa = '/Volumes/Macintosh_HD/Users/zg/bio/3_ensembF/3_scripts/2017_4_4/'
X = scipy.io.loadmat(pa+'/data/data_all_pickle/30/data.mat')['X'] # 30:breast cancer
y = scipy.io.loadmat(pa+'/data/data_all_pickle/30/data.mat')['y']
#X = scipy.io.loadmat(pa+'/data/data_all_pickle/11/data.mat')['X'] # 11:mesothelioma
#y = scipy.io.loadmat(pa+'/data/data_all_pickle/11/data.mat')['y']

n_clus = 3
clus = f_fuzzy_rwr_clusters(X, n_clus)
tfs = f_clus_to_tfs(clus, len(X))

y = y.astype(float)
#model = BaggingClassifier(base_estimator = tree.DecisionTreeClassifier(), \
#model = BaggingClassifier(base_estimator = svm.LinearSVR(), \
#model = BaggingClassifier(base_estimator = svm.LinearSVC(), \
model = BaggingClassifier(base_estimator = svm.SVC(kernel='linear'), \
    random_state=None, n_estimators = 100 )
    
[err_mat, models] = f_err_mat(X, y, clus, model)
"""



def f_err_2_decMat(err_mat, tfs, adv_whole=0.4, adv_self=0.5):
    '''
    Convert the err table to decision table. 
    
    '''
    dec_mat = np.zeros(( len(err_mat), err_mat[0].size-1 ), dtype=bool)
    
    # dec_ixs: for each instance, which clusters should be used.
    dec_ixs = []
    
    inst_clus = f_tfs_2_instClus(tfs)
    for i in range(0, len(err_mat)):
        
        # Matlab code:
        #dec_row = dec_mat(cur_nb_ix, :);
        #dec_row(:, end    ) = dec_row(:, end    ) - adv_whole;
        #dec_row(:, clus_id) = dec_row(:, clus_id) - adv_self;
        row = np.copy( err_mat[i, :] )
        #print row
        row[-1] = row[-1] - adv_whole
        
        inst_i_clus = inst_clus[i]
        if len(inst_i_clus) > 0:
            row[inst_i_clus] = row[inst_i_clus] - adv_self
        
        #print row
        ix_good_clus = list( np.where( row < row[-1] ) [0] )
        #print ix_good_clus
        if len(ix_good_clus) > 0:
            dec_mat[i, ix_good_clus] = True
            dec_ixs.append(ix_good_clus)
        else:
            dec_ixs.append([])
    
    return [dec_mat, dec_ixs]


#[dec_mat, dec_ixs] = f_err_2_decMat(err_mat, tfs)

def f_ICE_tr_te_all_clus(X_tr, X_te, clus, models, doNorm=True):
    '''
    Use the training data to predict the testing data. 
      Use whole training data to predict
      Use each cluster of training data to predict the testing data.
    '''
    y_pred_all = np.zeros(( len(X_te), len(clus) + 1 ))
    
    # the first col is the prediction using the whole data
    model_whole = models[-1]
    y_pred_all[:, 0] = f_te(X_te, model_whole)
    #y_pred_all[:, 0] = f_tr_te(X_tr, y_tr, X_te, model)
    #print 'whole model good '
    
    
    # start from the second col, the result is by each cluster
    for i in range(0, len(clus)):
        #Xi = X_tr[clus[i].flatten(), :]
        #yi = y_tr[clus[i].flatten() ]
        model_i = models[i]
        
        #model_a_clust = copy.deepcopy(model)
        
        try:
            y_pred_te = f_te(X_te, model_i)
        except :
            if model_i == 0:
                y_pred_te = np.zeros(len(X_te))
            elif model_i == 1:
                y_pred_te = np.ones(len(X_te))
            else:
                y_pred_te = np.zeros(len(X_te))
                y_pred_te.fill(np.nan)
                
        #except NotFittedError as e:
        #    print(repr(e))
        #    y_pred_te = np.zeros(len(X_te))
        #    y_pred_te.fill(np.nan)
        
        #print 'model '+str(i)+' good '
        #y_pred_te = f_tr_te(Xi, yi, X_te, model)
        
        if doNorm == True:
            templete = y_pred_all[:, 0]
            target = y_pred_te
            y_pred = f_quantileNorm(templete, target)
        else:
            y_pred = y_pred_te
        
        y_pred_all[:, i+1] = y_pred
    
    return y_pred_all

# test
#y_pred_all = f_ICE_tr_te_all_clus(X, X, clus, model)


def f_ICE_fit(X_tr, y_tr, n_clus, model, w=0.4, s=0.5):
    '''
    
    '''
    # rwr based fuzzy clustering
    clus = f_fuzzy_rwr_clusters(X_tr, n_clus)
    #print clus[0]
    tfs = f_clus_to_tfs(clus, len(X_tr))
    
    # train models and calculate the error-dicision tables
    y_tr = y_tr.astype(float)
    
    #model = BaggingClassifier(base_estimator = svm.SVC(kernel='linear'), \
    #    random_state=None, n_estimators = 100 )
        
    [err_mat, models] = f_err_mat(X_tr, y_tr, clus, model)
    
    [dec_mat, dec_ixs] = f_err_2_decMat(err_mat, tfs, w, s)
    print (' Done calucating decision table')
    return [clus, models, dec_ixs]


#def_deal_miss_v_1(d):
    '''
    deal with missing values by replacing them by mean.
    '''
    
    
def f_ICE_fit_2(X_tr, y_tr, n_clus, model, w=0.4, s=0.5):
    '''
    This version use the err mat to re-clustering
    '''
    # rwr based fuzzy clustering
    clus = f_fuzzy_rwr_clusters(X_tr, n_clus)
    #print clus[0]
    tfs = f_clus_to_tfs(clus, len(X_tr))
    
    # train models and calculate the error-dicision tables
    y_tr = y_tr.astype(float)
    
    #model = BaggingClassifier(base_estimator = svm.SVC(kernel='linear'), \
    #    random_state=None, n_estimators = 100 )
        
    [err_mat, models] = f_err_mat(X_tr, y_tr, clus, model)
    
    
    # ******************** re-clustering ********************
    n_iter = 2
    for i in range(0, n_iter):
        clus = f_fuzzy_rwr_clusters(err_mat, n_clus)
        tfs = f_clus_to_tfs(clus, len(X_tr))
        [err_mat, models] = f_err_mat(X_tr, y_tr, clus, model)
    # *******************************************************
    
    [dec_mat, dec_ixs] = f_err_2_decMat(err_mat, tfs, w, s)
    print (' Done calucating decision table')
    return [clus, models, dec_ixs]


def f_ICE_pred(X_tr, y_tr, X_te, clus, dec_ixs, models,N=5,alpha=1,beta=1):
    '''
    
    clus and inst_clus contains the same information that clus is the instances 
     ids for each cluster, while inst_clus stores that for each instance, which 
     cluster(s) it belongs to. 
     dec_ixs stores the good cluster(s) for each instance, which may include 
     even a remote cluster. each instance in dec_ixs does not contain the whole
     set of instances. 
    '''
    
    # the first col is the prediction using the whole data
    # start from the second col, the result is by each cluster
    y_pred_all = f_ICE_tr_te_all_clus(X_tr, X_te, clus, models)
    
    y_pred_ICE = np.zeros( len(X_te) )
    neighbour_mat = f_eu_dist2(X_tr, X_te)
    
    # ---------- for each testing instance ----------
    #n_partials = np.zeros( len(X_te) )
    #n_wholes = np.zeros( len(X_te) )
    
    for j in range(0, len(X_te) ):
        # for each testing instance
        # find the top 10 neighbors for each test instance
        neighbour_col = neighbour_mat[:, j].flatten()
        ix = np.argsort(neighbour_col )
        ix = ix[::-1]
        ix_top_neighbors = ix[0:N]
        #print 'testing inst ' + str(j)
        #print ' ix of top neighbors:'
        #print ix_top_neighbors
        
        # ---------- find all neighbors' picks ----------
        clus_ids_to_use = []
        nei_labels = []
        for cur_nb in range(0, N):
            # for each neighbour
            # find each neighbour's pick
            cur_nb_ix = ix_top_neighbors[cur_nb]
            clus_id_to_use = list( dec_ixs[cur_nb_ix] )
            
            clus_ids_to_use = clus_ids_to_use + clus_id_to_use
            
            # also find neighbor's label. maybe will be used later as KNN pred
            #  instead of using whole to pred.
            nei_labels = nei_labels + list( y_tr[cur_nb_ix] ) 
            
        
        #print ' clus_ids_to_use:'
        #print clus_ids_to_use
        
        # cluster id + 1 to make the ix fit the col id in y_pred_all
        a = clus_ids_to_use
        a = list( np.array(a) + 1 ) 
        clus_ids_to_use = a
        
        # number of partial models used 
        n_partial = len(clus_ids_to_use)
        # number of whole models used, based on parameters alpha, beta and N.
        n_whole = int( round( alpha*n_partial + beta*N ) )
        clus_ids_to_use = clus_ids_to_use + [0] * n_whole
        #print '   clus_ids_to_use:'
        #print clus_ids_to_use
        #print nei_labels
        
        y_pred_ICE[j] = np.nanmean(y_pred_all[j, clus_ids_to_use])
        
    print ('Done predicting testing instances.')
    return y_pred_ICE


# test
# pa = '/Volumes/Macintosh_HD/Users/zg/bio/3_ensembF/3_scripts/2017_4_4/'
# pa = '/Users/zg/Dropbox/bio/ICE_2018/'
# pa = './'
pa = 'C:/Users/zg/Dropbox/bio/ICE_2018/'

n_clus = 100
w = 0.4
s = 0.5
N = 5
alpha = 1
beta = 1

k_fold = 10

aucs_ICE = []
aucs_whole = []

# f_res = pa + 'data/res_ICE_bg_svm_1_iter.txt'
#f_res = pa + 'data/res_ICE_bg_svm_py.txt'
f_res = pa + 'data/res_ICE_SVM_py.txt'
f = open(f_res, 'w')

#for j in range(1, 50):
for j in range(1, 49):
    try:
        X = scipy.io.loadmat(pa+'data/data_all/'+str(j)+'/data.mat')['X'] # 30:breast cancer
        y = scipy.io.loadmat(pa+'data/data_all/'+str(j)+'/data.mat')['y']
        
        #X = scipy.io.loadmat(pa+'/data/data_all_pickle/30/data.mat')['X'] # 30:breast cancer
        #y = scipy.io.loadmat(pa+'/data/data_all_pickle/30/data.mat')['y']
        #X = scipy.io.loadmat(pa+'/data/data_all_pickle/37/data.mat')['X'] # 37:congress
        #y = scipy.io.loadmat(pa+'/data/data_all_pickle/37/data.mat')['y']
        
        #imgplot = plt.imshow(ori_graph, interpolation='nearest', aspect='auto')
        #plt.show()
        
        #sim = np.corrcoef(X)
        #np.fill_diagonal(sim, 0)
        
        #n_clus = 100
        
        #model = BaggingClassifier(base_estimator = svm.SVC(kernel='linear'), \
        #    random_state=None, n_estimators = 100 )
        model = svm.SVC(kernel='linear', probability = True)
        
        skf = StratifiedKFold(n_splits=k_fold)
        skf.get_n_splits(X, y)
        
        y_preds_ICE = np.zeros( y.size )
        y_preds_whole = np.zeros( y.size )
        
        fold_i = 1
        for train_index, test_index in skf.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_tr, X_te = X[train_index], X[test_index]
            y_tr, y_te = y[train_index], y[test_index]
            
            [clus, models, dec_ixs] = f_ICE_fit(X_tr, y_tr, n_clus, model, w, s)
            #[clus, models, dec_ixs] = f_ICE_fit_2(X_tr, y_tr, n_clus, model, w, s)
            
            y_pred_ICE = f_ICE_pred(X_tr, y_tr, X_te, clus, dec_ixs, models,N,alpha,beta)
            
            y_preds_ICE[test_index] = y_pred_ICE
            
            y_pred_whole = f_tr_te(X_tr, y_tr, X_te, model)
            y_preds_whole[test_index] = y_pred_whole
            
            print( j)
            print( 'fold ' + str(fold_i) + ' finished')
            fold_i = fold_i + 1
        
        
        auc_ICE = roc_auc_score(y.flatten(), y_preds_ICE.flatten() )
        auc_whole = roc_auc_score(y.flatten(), y_preds_whole.flatten() )
        
        print (auc_ICE, auc_whole)
        
        aucs_ICE.append(auc_ICE)
        aucs_whole.append(auc_whole)
        f.write(str(j) + '\t' + str(auc_ICE) + ' \t ' + str(auc_whole) + '\n')
    except:
        continue
        
        
        
        
        
        
        
        