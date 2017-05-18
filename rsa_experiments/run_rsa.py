from brainiak.matnormal import MatnormBRSA
from brainiak.reprsimil.brsa import BRSA
from brainiak.matnormal.covs import CovAR1, CovDiagonal
from sklearn.linear_model import LinearRegression
from brainiak.utils.utils import cov2corr
from scipy.io import savemat, loadmat
from pathlib import Path
from scipy import stats

import numpy as np


def run_experiment(par, input_path, outfile_path):
    mat = loadmat('%s/data_for_experiment.mat' % input_path)
    design = mat['design']
    n_nureg = par['n_nureg']
    if par['method'] == 'naive' and n_nureg > 10:
        # don't run naive multiple times, n_nureg isn't a thing
        return

    # if we have diff design by subj: 
    if design.ndim == 3:
        design = design[par['subj_num']]

    data = mat['fmri'][0][par['subj_num']]

    fname = "%s/results_%s_s%i_%inureg.mat" % (outfile_path, par['method'], par['subj_num'], par['n_nureg'])

    if Path(fname).exists():
        return

    if par['method'] == "naive":
        print("Running Naive RSA on subject %i!" % par['subj_num'])
        data = stats.zscore(data, axis=1, ddof=1)  
        m = LinearRegression(fit_intercept=False)
        m.fit(design, data.T)
        C = np.corrcoef(m.coef_.T)

    elif par['method'] == 'brsa':
        print("Running BRSA on subject %i!" % par['subj_num'])
        data = stats.zscore(data, axis=1, ddof=1)  
        m = BRSA(n_nureg=n_nureg)
        m.fit(X=data.T, design=design)
        C = cov2corr(m.U_)

    elif par['method'] == 'mnrsa':
        print("Running MNRSA on subject %i!" % par['subj_num'])
        # For mnrsa, zscore the whole thing but not by voxel
        # so that different voxels get to have different variances
        # but we don't blow out numerically
        data = stats.zscore(data, axis=None, ddof=1)  
        n_V, n_T = data.shape
        spacecov_model = CovDiagonal(size=n_V)
        timecov_model = CovAR1(size=n_T)
        model = MatnormBRSA(time_noise_cov=timecov_model,
                                space_noise_cov=spacecov_model,
                                optimizer='L-BFGS-B', n_nureg=n_nureg)
        model.fit(data.T, design)
        C = model.C_

    savemat(fname, {'C':C, 'method':par['method'], 'subject':par['subj_num']})

    return
