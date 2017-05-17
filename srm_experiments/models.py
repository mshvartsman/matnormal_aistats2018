from brainiak.funcalign.srm import SRM
from brainiak.matnormal.dpmnsrm import DPMNSRM
from brainiak.matnormal.covs import CovFullRankCholesky, CovAR1, CovDiagonal
from scipy import stats
from sklearn.decomposition import FastICA


# ECME not done for DPSRM since all updates are analytic
models = ['srm',
          'dpsrm_ecm',
          'dpsrm_orthos_ecm',
          'dpsrm_orthos_ecme',
          'dpmnsrm_ecm',
          'dpmnsrm_ecme',
          'dpmnsrm_orthos_ecm',
          'dpmnsrm_orthos_ecme',
          'ica']

def srm(train_data, test_data, n_features):
    # Z-score the data
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=1, ddof=1)

    model = SRM(n_iter=10, features=n_features)
    model.fit(train_data)
    projected_data = model.transform(test_data)
    for subject in range(n):
        projected_data[subject] = stats.zscore(projected_data[subject], axis=1, ddof=1)

    return projected_data

def ica(train_data, test_data, n_features):
    # Z-score the data
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=1, ddof=1)

    model = FastICA(n_components=n_features)
    model.fit(train_data)
    projected_data = model.transform(test_data)
    for subject in range(n):
        projected_data[subject] = stats.zscore(projected_data[subject], axis=1, ddof=1)

    return projected_data


def dpsrm_ecm(train_data, test_data, n_features):
    # Z-score the data since we're not modeling noise variance
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=1, ddof=1)

    model = DPMNSRM(n_features=n_features)
    model.fit(train_data, max_iter=25, convergence_tol=1e-5)
    projected_data = model.transform(test_data)
    
    # if SRM does it maybe we do it too? still no idea why it should help
    for subject in range(n):
        projected_data[subject] = stats.zscore(projected_data[subject], axis=1, ddof=1)

    return projected_data


def dpsrm_orthos_ecm(train_data, test_data, n_features):
    # Z-score the data since we're not modeling noise variance
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=1, ddof=1)

    model = DPMNSRM(n_features=n_features, s_constraint="ortho", w_cov=CovFullRankCholesky)
    model.fit(train_data, max_iter=25, convergence_tol=1e-5)
    projected_data = model.transform(test_data)
    return projected_data


def dpmnsrm_ecm(train_data, test_data, n_features):

    n = len(train_data)
    # no zscoring here, let the model figure out the informative voxels
    # but do rescale so that we don't get numeric problems
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=None, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=None, ddof=1)

    model = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1)
    model.fit(train_data, max_iter=25, convergence_tol=1e-5)
    projected_data = model.transform(test_data)
    
    # if SRM does it maybe we do it too? still no idea why it should help
    for subject in range(n):
        projected_data[subject] = stats.zscore(projected_data[subject], axis=1, ddof=1)

    return projected_data

def dpmnsrm_orthos_ecm(train_data, test_data, n_features):

    n = len(train_data)
    # no zscoring here, let the model figure out the informative voxels
    # but do rescale so that we don't get numeric problems
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=None, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=None, ddof=1)


    model = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1, s_constraint="ortho", w_cov=CovFullRankCholesky)
    model.fit(train_data, max_iter=25, convergence_tol=1e-5)
    projected_data = model.transform(test_data)
    
    # if SRM does it maybe we do it too? still no idea why it should help
    for subject in range(n):
        projected_data[subject] = stats.zscore(projected_data[subject], axis=1, ddof=1)

    return projected_data

def dpsrm_orthos_ecme(train_data, test_data, n_features):
    # Z-score the data since we're not modeling noise variance
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=1, ddof=1)

    model = DPMNSRM(n_features=n_features, s_constraint="ortho", w_cov=CovFullRankCholesky, algorithm="ECME")
    model.fit(train_data, max_iter=25, convergence_tol=1e-5)
    projected_data = model.transform(test_data)
    return projected_data


def dpmnsrm_ecme(train_data, test_data, n_features):

    n = len(train_data)
    # no zscoring here, let the model figure out the informative voxels
    # but do rescale so that we don't get numeric problems
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=None, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=None, ddof=1)


    model = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1, algorithm="ECME")
    model.fit(train_data, max_iter=25, convergence_tol=1e-5)
    projected_data = model.transform(test_data)
    
    # if SRM does it maybe we do it too? still no idea why it should help
    for subject in range(n):
        projected_data[subject] = stats.zscore(projected_data[subject], axis=1, ddof=1)

    return projected_data

def dpmnsrm_orthos_ecme(train_data, test_data, n_features):

    n = len(train_data)
    # no zscoring here, let the model figure out the informative voxels
    # but do rescale so that we don't get numeric problems
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=None, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=None, ddof=1)


    model = DPMNSRM(n_features=n_features, space_noise_cov=CovDiagonal, time_noise_cov=CovAR1, s_constraint="ortho", w_cov=CovFullRankCholesky, algorithm="ECME")
    model.fit(train_data, max_iter=25, convergence_tol=1e-5)
    projected_data = model.transform(test_data)
    
    # if SRM does it maybe we do it too? still no idea why it should help
    for subject in range(n):
        projected_data[subject] = stats.zscore(projected_data[subject], axis=1, ddof=1)

    return projected_data


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
