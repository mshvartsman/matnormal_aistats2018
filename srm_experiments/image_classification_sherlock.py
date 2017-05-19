## heavily based on SRM brainiak example
import scipy.io
import os
from sklearn.svm import NuSVC, SVC
import numpy as np
import models
from pathlib import Path
import pickle
import logging
import pandas as pd
from collections import OrderedDict
from itertools import product
logging.basicConfig(level=logging.INFO)

input_path = '/home/ms44/nips2017_data/sherlock'
outfile_path = '/home/ms44/nips2017_data/sherlock/results'

# input_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/sherlock'
# outfile_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/sherlock/results'

def run_experiment(par):
    model = getattr(models, par['model'])

    # Load the input data that contains the movie stimuli for unsupervised training with SRM
    movie_data = np.load('%s/sherlock_pmc_813vox.npy'%input_path)
    movie_data = movie_data.astype(np.float64, copy=True)
    subjects = movie_data.shape[0]
    nVoxel, nTR = movie_data[0].shape

    recall_data = pickle.load(open("%s/recall_data_all.pickle" % input_path, 'rb'))

    transformed_data = model(movie_data, recall_data, par['features'])
    # transpose now -- we need it like that for the classifier, and it makes
    # array wrangling easier
    transformed_data = np.array([td.T for td in transformed_data])
    # Read the labels of the image data for training the classifier.
    labels = np.array(pickle.load(open("%s/label.pickle" % input_path, 'rb')))

    # Run a leave-one-out cross validation with the subjects
    accuracy = np.zeros((subjects,))
    train_acc = np.zeros((subjects,)) # upper bound

    for subject in range(subjects):
        
        # test_labels = labels

        # Concatenate the subjects' data for training into one matrix
        train_subjects = list(range(subjects))
        train_subjects.remove(subject)
        train_labels = np.concatenate(labels[train_subjects], 0)
        test_labels = labels[subject]
        # train_data = np.zeros(recall_data[0].shape[0], np.sum(test_obs_per_subj[train_subjects]))
        train_data = np.concatenate(transformed_data[train_subjects], 0)
        test_data = transformed_data[subject]

        # Train a Nu-SVM classifier using scikit learn
        # classifier = NuSVC(nu=0.1, kernel='linear')
        classifier = SVC(kernel='linear')
        classifier = classifier.fit(train_data, train_labels)
        predicted_labels = classifier.predict(test_data)
        accuracy[subject] = sum(predicted_labels == test_labels)/float(len(predicted_labels))
        train_acc[subject] = sum(classifier.predict(train_data) == train_labels)/float(len(train_labels))

    return {'model': par['model'],
            'features': par['features'],
             'mean_acc': np.mean(accuracy),
             'training_mean_acc': np.mean(train_acc),
             'std_acc': np.std(accuracy),
             'std_train_acc': np.std(train_acc)}

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
        ('features',  [10, 30, 50])])

    # cartesian over param settings
    allpar = [dict(parset) for parset in (zip(runPars.keys(), p)
              for p in product(*runPars.values()))]
    allpar = [dict(parset) for parset in (zip(runPars.keys(), p)
              for p in product(*runPars.values()))]

    pointsPerId = len(allpar) / totalIDs
    start = int((myID-1)*pointsPerId)
    end = int(len(allpar) if myID == totalIDs else (myID)*pointsPerId)
    print("Doing Params %s to %s (inclusive)" % (start, end-1))
    # mypar = allpar[start:end]

    for parnum in range(start, end):
        mypar = allpar[parnum]        
        fname = '%s/results_%s_%ifeatures.csv' % (outfile_path, mypar['model'], mypar['features'])
        if Path(fname).exists(): 
            print("Found %s, skipping" % fname)
            continue 
        else:
            res = pd.DataFrame([run_experiment(allpar[parnum])])
            res.to_csv(fname)

    print("Done!")

