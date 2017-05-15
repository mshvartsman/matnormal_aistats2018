from brainiak.funcalign.srm import SRM
from brainiak.matnormal.dpmnsrm import DPMNSRM
from scipy import stats

models = ['srm', 'dpsrm']

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


def dpsrm(train_data, test_data, n_features):
    # Z-score the data since we're not modeling noise variance
    n = len(train_data)
    for subject in range(n):
        train_data[subject] = stats.zscore(train_data[subject], axis=1, ddof=1)
        test_data[subject] = stats.zscore(test_data[subject], axis=1, ddof=1)

    model = DPMNSRM(n_features=n_features)
    model.fit(train_data, max_iter=10)
    projected_data = model.transform(test_data)
    # do not zscore outputs again though (why do this?)
    return projected_data

