from brainiak.funcalign.srm import SRM
from brainiak.matnormal.dpmnsrm import DPMNSRM
from brainiak.matnormal.covs import CovFullRankCholesky, CovAR1, CovDiagonal
from sklearn.decomposition import FastICA, PCA
from scipy import stats
import numpy as np


# ECME not done for DPSRM since all updates are analytic
models = ['srm',
          'dpsrm_ecm',
          'dpmnsrm_ecm',
          'pca','ica']
          # 'dpmnsrm_ecme']

def dpmnsrm_wposterior(X, model):
    S = model.s_
    tcov = model.time_cov.Sigma.eval(session=model.sess)
    b = np.average(X, 1)[:,None]
    k = S.shape[0]
    wprec_prime = np.eye(k) + S @ np.linalg.solve(tcov, S.T)
    w_prime = (X-b) @ np.linalg.solve(tcov, np.linalg.solve(wprec_prime, S).T )
    return w_prime


def rmse(x, xtrue):
    return np.sqrt(np.average((x-xtrue)**2))

def relative_mse(x, xtrue):
    return np.sum((x-xtrue)**2) / np.sum(xtrue**2)


def srm(train_data, test_data, n_features):
    # Z-score the data
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)

    test_data = stats.zscore(test_data, axis=1, ddof=1)
    model = SRM(n_iter=10, features=n_features)
    model.fit(train_data)
    # now we do the mstep of srm to get its W for the new X
    a_new = test_data.dot(model.s_.T)
    perturbation = np.zeros(a_new.shape)
    np.fill_diagonal(perturbation, 0.001)
    u, s, v = np.linalg.svd(a_new + perturbation, full_matrices=False)
    w_new = u.dot(v)

    return rmse(w_new.dot(model.s_), test_data), relative_mse(w_new.dot(model.s_), test_data)

def dpsrm_ecm(train_data, test_data, n_features):
    # Z-score the data since we're not modeling noise variance
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)

    test_data = stats.zscore(test_data, axis=1, ddof=1)

    model = DPMNSRM(n_features=n_features)
    model.fit(train_data, max_iter=10, convergence_tol=1e-5)

    w_new = dpmnsrm_wposterior(test_data, model)

    return rmse(w_new.dot(model.s_), test_data), relative_mse(w_new.dot(model.s_), test_data)

def dpmnsrm_ecm(train_data, test_data, n_features):

    n = len(train_data)
    # no zscoring here, let the model figure out the informative voxels
    # but do rescale so that we don't get numeric problems
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=None, ddof=1)

    test_data = stats.zscore(test_data, axis=None, ddof=1)

    model = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1)
    model.fit(train_data, max_iter=10, convergence_tol=1e-5)
    w_new = dpmnsrm_wposterior(test_data, model)

    return rmse(w_new.dot(model.s_), test_data), relative_mse(w_new.dot(model.s_), test_data)

def dpmnsrm_ecme(train_data, test_data, n_features):

    n = len(train_data)
    # no zscoring here, let the model figure out the informative voxels
    # but do rescale so that we don't get numeric problems
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=None, ddof=1)

    test_data = stats.zscore(test_data, axis=None, ddof=1)

    model = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1, algorithm="ECME")
    model.fit(train_data, max_iter=10, convergence_tol=1e-5)
    w_new = dpmnsrm_wposterior(test_data, model)

    return rmse(w_new.dot(model.s_), test_data), relative_mse(w_new.dot(model.s_), test_data)


def pca(train_data, test_data, n_features):
    # Z-score the data
    n, v, t = len(train_data), train_data[0].shape[0],  train_data[0].shape[1]
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)

    model = PCA(n_components=n_features)
    s = model.fit_transform(np.reshape(train_data, (n*v, t)).T)
    w_new = np.linalg.lstsq(s, test_data.T)[0].T

    return rmse(w_new.dot(s.T), test_data), relative_mse(w_new.dot(s.T), test_data)


def ica(train_data, test_data, n_features):
    # Z-score the data
    n, v, t = len(train_data), train_data[0].shape[0],  train_data[0].shape[1]
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)

    model = FastICA(n_components=n_features)
    s = model.fit_transform(np.reshape(train_data, (n*v, t)).T)
    w_new = np.linalg.lstsq(s, test_data.T)[0].T

    return rmse(w_new.dot(s.T), test_data), relative_mse(w_new.dot(s.T), test_data)


# def hyperalignment(train_data, test_data, n_features):
#     from mvpa2.suite import Hyperalignment
#     from mvpa2.datasets import Dataset
#     # Z-score the data
#     n = len(train_data)
#     for subject in range(n):
#         train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)
#         test_data[subject] = stats.zscore(test_data[subject], axis=1, ddof=1)

#     ds_train = [Dataset(td) for td in train_data]
#     ds_test = [Dataset(td) for td in test_data]
#     ha = Hyperalignment()
#     ha.train(ds_train)
#     projected_data = ha(ds_test)
#     for subject in range(n):
#         projected_data[subject] = stats.zscore(projected_data[subject], axis=1, ddof=1)

#     return projected_data
