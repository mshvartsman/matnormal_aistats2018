import os
from brainiak.matnormal import MatnormBRSA
from brainiak.reprsimil.brsa import BRSA
from brainiak.matnormal.covs import CovAR1, CovDiagonal
from sklearn.linear_model import LinearRegression
from brainiak.utils.utils import cov2corr
from scipy.io import savemat, loadmat
from collections import OrderedDict
from pathlib import Path
from itertools import product
from scipy import stats

import numpy as np

input_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/resting_state/'
outfile_template = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider/results_%s_s%i.csv'


def run_experiment(par):
    mat = loadmat('%s/data_for_experiment.mat' % input_path)
    design = mat['design']
    data = mat['fmri'][0][par['subj_num']]

    fname = outfile_template % (par['method'], par['subj_num'])

    if Path(fname).exists():
        return

    if par['method'] == "naive":
        data = stats.zscore(data, axis=1, ddof=1)  
        m = LinearRegression(fit_intercept=False)
        m.fit(design, data.T)
        C = np.corrcoef(m.coef_.T)
    elif par['method'] == 'brsa':
        data = stats.zscore(data, axis=1, ddof=1)  
        m = BRSA(n_nureg=15)
        m.fit(X=data.T, design=design)
        C = cov2corr(m.U_)
    elif par['method'] == 'mnrsa':
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


if __name__ == "__main__":

    try:
        myID = int(os.environ["SLURM_ARRAY_TASK_ID"])
        totalIDs = int(os.environ["SLURM_ARRAY_TASK_MAX"])
    except KeyError:
        myID = 1
        totalIDs = 1

    print("Job %s of %s reporting in!" % (myID, totalIDs))

    runPars = OrderedDict([
        ('method', ['naive','brsa','mnrsa']),
        ('subj_num',  np.arange(29))])

    # cartesian over param settings
    allpar = [dict(parset) for parset in (zip(runPars.keys(), p)
              for p in product(*runPars.values()))]

    pointsPerId = len(allpar) / totalIDs
    start = int((myID-1)*pointsPerId)
    end = int(len(allpar) if myID == totalIDs else (myID)*pointsPerId)
    print("Doing Params %s to %s (inclusive)" % (start, end-1))

    for parnum in range(start, end):
        run_experiment(allpar[parnum])

    print("Done!")
