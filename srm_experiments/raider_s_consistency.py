## heavily based on SRM brainiak example
import scipy.io
import os
import numpy as np
from pathlib import Path
import models_loo_reconstruction as models
import logging
import pandas as pd
from collections import OrderedDict
from itertools import product
logging.basicConfig(level=logging.INFO)

input_path = '/home/ms44/nips2017_data/raider'
outfile_path = '/home/ms44/nips2017_data/raider/results_s_consistency'

# input_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider'
# outfile_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider/results_loo_recon'

def run_experiment(par):
    model = getattr(models, par['model'])
    
    # Load the input data that contains the movie stimuli for unsupervised training with SRM
    movie_file = scipy.io.loadmat('%s/movie_data.mat' % input_path)
    movie_data_left = movie_file['movie_data_lh']
    movie_data_right = movie_file['movie_data_rh']
    subjects = movie_data_left.shape[2]

    # Convert data to a list of arrays matching SRM input.
    # Each element is a matrix of voxels by TRs.
    # Also, concatenate data from both hemispheres in the brain.
    train_data = []
    for s in range(subjects):
        train_data.append(np.concatenate([movie_data_left[:, :, s], movie_data_right[:, :, s]], axis=0))

    consistency_norm = model(train_data, par['features'])

    return {'model': par['model'],
            'features': par['features'],
            'cnorm': consistency_norm, 
            'fold': par['fold']}

if __name__ == "__main__":

    try:
        myID = int(os.environ["SLURM_ARRAY_TASK_ID"])
        totalIDs = int(os.environ["SLURM_ARRAY_TASK_MAX"])
    except KeyError:
        myID = 1
        totalIDs = 1

    print("Job %s of %s reporting in!" % (myID, totalIDs))

    runPars = OrderedDict([
        ('model', models.models),
        ('features',  [10, 30, 50]),
        ('fold', np.arange(10))])

    # cartesian over param settings
    allpar = [dict(parset) for parset in (zip(runPars.keys(), p)
              for p in product(*runPars.values()))]

    pointsPerId = len(allpar) / totalIDs
    start = int((myID-1)*pointsPerId)
    end = int(len(allpar) if myID == totalIDs else (myID)*pointsPerId)
    print("Doing Params %s to %s (inclusive)" % (start, end-1))
    # mypar = allpar[start:end]

    for parnum in range(start, end):
        mypar = allpar[parnum]        
        fname = '%s/results_s_consistency_%s_%ifeatures_fold%i.csv' % (outfile_path, mypar['model'], mypar['features'], mypar['fold'])
        if Path(fname).exists(): 
            print("Found %s, skipping" % fname)
            continue
        else:
            res = pd.DataFrame([run_experiment(allpar[parnum])])
            res.to_csv(fname)

    print("Done!")

