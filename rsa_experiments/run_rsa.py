from brainiak.matnormal import MatnormBRSA
from brainiak.reprsimil.brsa import BRSA
from brainiak.matnormal.covs import CovAR1, CovDiagonal
from sklearn.linear_model import LinearRegression
from brainiak.utils.utils import cov2corr
from scipy.io import savemat, loadmat
from pathlib import Path
from scipy import stats

import numpy as np


def run_experiment(par, input_path, outfile_template):
    mat = loadmat('%s/data_for_experiment.mat' % input_path)
    design = mat['design']

    # if we have diff design by subj: 
    if design.ndim == 3:
        design = design[par['subj_num']]

    data = mat['fmri'][0][par['subj_num']]

    fname = outfile_template % (par['method'], par['subj_num'])

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
        m = BRSA(n_nureg=15)
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
                                optimizer='L-BFGS-B', n_nureg=15)
        model.fit(data.T, design)
        C = model.C_

    savemat(fname, {'C':C, 'method':par['method'], 'subject':par['subj_num']})

    return
