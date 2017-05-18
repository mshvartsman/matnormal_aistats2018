import os
from collections import OrderedDict
from itertools import product
import numpy as np
from run_rsa import run_experiment

input_path = '/home/ms44/nips2017_data/resting_state'
outfile_path = '/home/ms44/nips2017_data/resting_state/results/'

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
        ('subj_num',  np.arange(29)), 
        ('n_nureg',  [10, 20, 50])])

    # cartesian over param settings
    allpar = [dict(parset) for parset in (zip(runPars.keys(), p)
              for p in product(*runPars.values()))]

    pointsPerId = len(allpar) / totalIDs
    start = int((myID-1)*pointsPerId)
    end = int(len(allpar) if myID == totalIDs else (myID)*pointsPerId)
    print("Doing Params %s to %s (inclusive)" % (start, end-1))

    for parnum in range(start, end):
        mypar = allpar[parnum]
        run_experiment(allpar[parnum], input_path, outfile_path)

    print("Done!")
