# With profuse thanks to Cameron Chen for the preprocessed data and pipeline! 

input_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/'

import scipy.io
from scipy.stats import stats
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import NuSVC
import numpy as np
import brainiak.funcalign.srm
from scipy.stats import norm
from brainiak.matnormal.dpsrm import DPSRM
from brainiak.matnormal.srm_margw_em import DPMNSRM
from brainiak.funcalign.srm import SRM
from brainiak.matnormal.dpmnsrm_orthos import DPMNSRM_OrthoS
from brainiak.matnormal.covs import CovAR1, CovDiagonal, CovFullRankCholesky
import logging
logging.basicConfig(level=logging.DEBUG)

import sys

def timesegmentmatching_accuracy_evaluation_loo_cv(data, win_size=6, method=""):
 
    nsubjs = len(data)
    (ndim, nsample) = data[0].shape
    accu = np.zeros(shape=nsubjs)

    nseg = nsample - win_size 
    # mysseg prediction prediction
    trn_data = np.zeros((ndim*win_size, nseg))

    # the trn data also include the tst data, but will be subtracted when 
    # calculating A
    for m in range(nsubjs):
        for w in range(win_size):
            trn_data[w*ndim:(w+1)*ndim,:] += data[m][:,w:(w+nseg)]

    for tst_subj in range(nsubjs):
        tst_data = np.zeros((ndim*win_size, nseg))
        for w in range(win_size):
            tst_data[w*ndim:(w+1)*ndim,:] = data[tst_subj][:,w:(w+nseg)]

        A = stats.zscore((trn_data - tst_data),axis=0, ddof=1)
        B = stats.zscore(tst_data, axis=0, ddof=1)

        corr_mtx = B.T.dot(A)

        for i in range(nseg):
            for j in range(nseg):
                if abs(i-j)<win_size and i != j :
                    corr_mtx[i,j] = -np.inf

        rank = np.argmax(corr_mtx, axis=1)
        accu[tst_subj] = sum(rank == range(nseg)) / float(nseg)
    print(accu)
    print("%s: The average accuracy among all subjects is %.4f +/- %.4f" % (method, np.mean(accu), np.std(accu)))

if __name__ == "__main__":
    
    movie_data = np.stack(scipy.io.loadmat('%s/sherlock_pmc.mat'%input_path)['movie_all'][:,0])
    movie_data = movie_data.astype(np.float64, copy=True)
    subjects = movie_data.shape[0]
    nVoxel, nTR = movie_data[0].shape

    train_data = []
    test_data = []
    
    for s in range(subjects):
        train_data.append(movie_data[s,:,:nTR//2])
        test_data.append(movie_data[s,:,nTR//2:])

    for subject in range(subjects):
        train_data[subject] = stats.zscore(train_data[subject],axis=1,ddof=1)
    for subject in range(subjects):
        test_data[subject] = stats.zscore(test_data[subject],axis=1,ddof=1)

    srm = SRM(n_iter=10, features=50)
    srm.fit(train_data)

    data_shared = srm.transform(test_data)

    timesegmentmatching_accuracy_evaluation_loo_cv(data_shared, win_size=6, method="SRM")

    dpsrm = DPSRM(n_features=50)
    dpsrm.fit(np.array(train_data))
    data_shared_dpsrm = dpsrm.transform(test_data)    
    timesegmentmatching_accuracy_evaluation_loo_cv(data_shared_dpsrm, win_size=6, method="DPSRM")

    data_shared_dpsrm_orthow = dpsrm.transform_orthow(test_data)
    timesegmentmatching_accuracy_evaluation_loo_cv(data_shared_dpsrm_orthow, win_size=6, method="DPSRM-Ortho")

    dpmnsrm = DPMNSRM(n_features=50, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1)
    dpmnsrm.fit(np.array(train_data))
    data_shared_dpmnsrm = dpmnsrm.transform(test_data)    
    timesegmentmatching_accuracy_evaluation_loo_cv(data_shared_dpmnsrm, win_size=6, method="DPMNSRM")

    data_shared_dpmnsrm_orthow = dpmnsrm.transform_orthow(test_data)
    timesegmentmatching_accuracy_evaluation_loo_cv(data_shared_dpmnsrm_orthow, win_size=6, method="DPMNSRM-Ortho")

    dpmnsrm_orthos = DPMNSRM_OrthoS(n_features=50, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1, w_cov=CovFullRankCholesky)
    dpmnsrm_orthos.fit(np.array(train_data))
    data_shared_dpmnsrm_orthos = dpmnsrm_orthos.transform(test_data)    
    timesegmentmatching_accuracy_evaluation_loo_cv(data_shared_dpmnsrm_orthos, win_size=6, method="DPMNSRM-OrthoS")
