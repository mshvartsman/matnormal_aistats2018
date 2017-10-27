## heavily based on SRM brainiak example
import scipy.io
import os
from sklearn.svm import NuSVC
import numpy as np
from pathlib import Path
import models
import logging
import pandas as pd
from collections import OrderedDict
from itertools import product
from brainiak.matnormal.dpmnsrm import DPMNSRM
from brainiak.matnormal.covs import CovIdentity, CovDiagonal, CovAR1#, CovIncompleteCholeskyPlusDiag
from scipy import stats
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
logging.basicConfig(level=logging.INFO)


input_path = '/home/ms44/nips2017_data/raider'
outfile_path = '/home/ms44/nips2017_data/raider/results'


covs = [CovIdentity, CovDiagonal, CovAR1]#], CovIncompleteCholeskyPlusDiag]
space_covs = hp.choice('space_cov', covs)
time_covs = hp.choice('time_cov', covs)
s_constraints = hp.choice('s_constraint', ['gaussian', 'ortho'])
algos = hp.choice('algo', ['ECM', 'ECME'])
features = hp.qlognormal('features', 3, 1, 1)
search_graph = {'space_cov': space_covs,
                'time_cov': time_covs,
                's_constraint': s_constraints,
                'algo': algos,
                'features': features}

# input_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider'
# outfile_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider/results'
# par = {'model':'ica', 'n_features':10}


def objective(x):
    print(x)
    data = np.load('raider_inp.npz')
    movie_data = data['movie_data']
    image_data = data['image_data']

    n = len(movie_data)
    # no zscoring here, let the model figure out the informative voxels
    # but do rescale so that we don't get numeric problems
    for subject in range(n):
        movie_data[subject] = stats.zscore(movie_data[subject], axis=None, ddof=1)
        image_data[subject] = stats.zscore(image_data[subject], axis=None, ddof=1)
    subjects = len(movie_data)

    model = DPMNSRM(n_features=np.int(x['features']), space_noise_cov=x['space_cov'],
                    time_noise_cov=x['time_cov'], s_constraint=x['s_constraint'], algorithm=x['algo'])
    model.fit(movie_data, max_iter=25, convergence_tol=1e-5)
    projected_data = model.transform(image_data)
    
    # if SRM does it maybe we do it too? still no idea why it should help
    for subject in range(n):
        projected_data[subject] = stats.zscore(projected_data[subject], axis=1, ddof=1)

    # Read the labels of the image data for training the classifier.
    labels = scipy.io.loadmat('%s/label.mat' % input_path)
    labels = np.squeeze(labels['label'])

    # Run a leave-one-out cross validation with the subjects
    train_labels = np.tile(labels, subjects-1)
    test_labels = labels
    accuracy = np.zeros((subjects,))
    train_acc  = np.zeros((subjects,)) # upper bound

    for subject in range(subjects):
        # Concatenate the subjects' data for training into one matrix
        train_subjects = list(range(subjects))
        train_subjects.remove(subject)
        TRs = projected_data[0].shape[1]
        movie_data = np.zeros((projected_data[0].shape[0], len(train_labels)))

        for train_subject in range(len(train_subjects)):
            start_index = train_subject*TRs
            end_index = start_index+TRs
            movie_data[:, start_index:end_index] = projected_data[train_subjects[train_subject]]

        # Train a Nu-SVM classifier using scikit learn
        classifier = NuSVC(nu=0.5, kernel='linear')
        classifier = classifier.fit(movie_data.T, train_labels)
        predicted_labels = classifier.predict(projected_data[subject].T)
        accuracy[subject] = sum(predicted_labels == test_labels)/float(len(predicted_labels))
        train_acc[subject] = sum(classifier.predict(movie_data.T) == train_labels)/float(len(train_labels))

    return {'loss':-np.mean(accuracy),
             'training_mean_acc': np.mean(train_acc),
             'mean_acc': np.mean(accuracy),
             'std_acc': np.std(accuracy),
             'std_train_acc': np.std(train_acc),
             'status': STATUS_OK}

trials = Trials()
best = fmin(objective,
    space=search_graph,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials)

print(best)