from brainiak.funcalign.srm import SRM
from brainiak.matnormal.dpmnsrm import DPMNSRM
from brainiak.matnormal.covs import CovFullRankCholesky, CovAR1, CovDiagonal
from scipy import stats
import numpy as np


# ECME not done for DPSRM since all updates are analytic
models = ['srm',
          'dpsrm_ecm',
          'dpmnsrm_ecm']
          # 'dpmnsrm_ecme']

def norm_from_ortho(s1, s2):
    R = np.linalg.lstsq(s1.T,s2.T)[0]
    return np.linalg.norm(R.T @ R - np.eye(R.shape[0]))


def srm(train_data, n_features):
    # Z-score the data
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)

    all_subj = np.arange(n)
    np.random.shuffle(all_subj)

    model1 = SRM(n_iter=10, features=n_features)
    model2 = SRM(n_iter=10, features=n_features)

    dset1 = train_data[all_subj[:n//2],:,:]
    dset2 = train_data[all_subj[n//2:],:,:]

    model1.fit(dset1)
    model2.fit(dset2)
    return norm_from_ortho(model1.s_, model2.s_)

def dpsrm_ecm(train_data, test_data, n_features):
    # Z-score the data since we're not modeling noise variance
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)

    all_subj = np.arange(n)
    np.random.shuffle(all_subj)

    model1 = DPMNSRM(n_features=n_features)
    model2 = DPMNSRM(n_features=n_features)

    dset1 = train_data[all_subj[:n//2],:,:]
    dset2 = train_data[all_subj[n//2:],:,:]

    model1.fit(dset1)
    model2.fit(dset2)
    return norm_from_ortho(model1.s_, model2.s_)

def dpmnsrm_ecm(train_data, test_data, n_features):

    n = len(train_data)
    # no zscoring here, let the model figure out the informative voxels
    # but do rescale so that we don't get numeric problems
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=None, ddof=1)

    all_subj = np.arange(n)
    np.random.shuffle(all_subj)

    model1 = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1)
    model2 = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1)

    dset1 = train_data[all_subj[:n//2],:,:]
    dset2 = train_data[all_subj[n//2:],:,:]

    model1.fit(dset1)
    model2.fit(dset2)
    return norm_from_ortho(model1.s_, model2.s_)
