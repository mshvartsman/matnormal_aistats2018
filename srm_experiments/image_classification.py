## heavily based on SRM brainiak example

input_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/brainiak_example_data'

import scipy.io
from scipy.stats import stats
from sklearn.metrics import confusion_matrix
from sklearn.svm import NuSVC, SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import brainiak.funcalign.srm
from brainiak.matnormal.dpsrm import DPSRM
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Load the input data that contains the movie stimuli for unsupervised training with SRM
    movie_file = scipy.io.loadmat('%s/movie_data.mat' % input_path)
    movie_data_left = movie_file['movie_data_lh']
    movie_data_right = movie_file['movie_data_rh']
    subjects = movie_data_left.shape[2]

    # Convert data to a list of arrays matching SRM input.
    # Each element is a matrix of voxels by TRs.
    # Also, concatenate data from both hemispheres in the brain.

    movie_data = []
    for s in range(subjects):
        movie_data.append(np.concatenate([movie_data_left[:, :, s], movie_data_right[:, :, s]], axis=0))

    # Z-score the data
    for subject in range(subjects):
        movie_data[subject] = stats.zscore(movie_data[subject], axis=1, ddof=1)

    # Run SRM with the movie data
    srm = brainiak.funcalign.srm.SRM(n_iter=10, features=50)
    srm.fit(movie_data)

    # Run DPSRM with the same data
    dpsrm = DPSRM(n_features=50)
    dpsrm.fit(np.array(movie_data), n_iter=10)

    # Load the input data that contains the image stimuli and its labels for training a classifier
    image_file = scipy.io.loadmat('%s/image_data.mat' % input_path)
    image_data_left = image_file['image_data_lh']
    image_data_right = image_file['image_data_rh']

    # Convert data to a list of arrays matching SRM input.
    # Each element is a matrix of voxels by TRs.
    # Also, concatenate data from both hemispheres in the brain.
    image_data = []
    for s in range(subjects):
        image_data.append(np.concatenate([image_data_left[:, :, s], image_data_right[:, :, s]], axis=0))

    assert image_data[0].shape[0] == movie_data[0].shape[0], "Number of voxels in movie data and image data do not match!"

    # Z-score the image data
    for subject in range(subjects):
        image_data[subject] = stats.zscore(image_data[subject], axis=1, ddof=1)

    # Z-score the shared response data
    image_data_shared = srm.transform(image_data)
    for subject in range(subjects):
        image_data_shared[subject] = stats.zscore(image_data_shared[subject], axis=1, ddof=1)

    # project to shared space using dpsrm and zscore
    image_data_shared_dpsrm = dpsrm.transform(image_data)
    for subject in range(subjects):
        image_data_shared_dpsrm[subject] = stats.zscore(image_data_shared_dpsrm[subject], axis=1, ddof=1)

    # project to shared space using dpsrm with orthogonalized W and zscore
    image_data_shared_dpsrm_orthow = dpsrm.transform_orthow(image_data)
    for subject in range(subjects):
        image_data_shared_dpsrm_orthow[subject] = stats.zscore(image_data_shared_dpsrm_orthow[subject], axis=1, ddof=1)


    # Read the labels of the image data for training the classifier.
    labels = scipy.io.loadmat('%s/label.mat' % input_path)
    labels = np.squeeze(labels['label'])

    # Run a leave-one-out cross validation with the subjects
    train_labels = np.tile(labels, subjects-1)
    test_labels = labels
    accuracy = np.zeros((subjects,))
    accuracy_dpsrm = np.zeros((subjects,))
    accuracy_dpsrm_orthow = np.zeros((subjects,))
    probassigned = np.zeros((subjects,))
    probassigned_dpsrm = np.zeros((subjects,))
    probassigned_dpsrm_orthow = np.zeros((subjects,))

    for subject in range(subjects):
        # Concatenate the subjects' data for training into one matrix
        train_subjects = list(range(subjects))
        train_subjects.remove(subject)
        TRs = image_data_shared[0].shape[1]
        train_data = np.zeros((image_data_shared[0].shape[0], len(train_labels)))
        train_data_dpsrm = np.zeros((image_data_shared_dpsrm[0].shape[0], len(train_labels)))
        train_data_dpsrm_orthow = np.zeros((image_data_shared_dpsrm[0].shape[0], len(train_labels)))
        for train_subject in range(len(train_subjects)):
            start_index = train_subject*TRs
            end_index = start_index+TRs
            train_data[:, start_index:end_index] = image_data_shared[train_subjects[train_subject]]
            train_data_dpsrm[:, start_index:end_index] = image_data_shared_dpsrm[train_subjects[train_subject]]
            train_data_dpsrm_orthow[:, start_index:end_index] = image_data_shared_dpsrm_orthow[train_subjects[train_subject]]

        # Train a Nu-SVM classifier using scikit learn
        classifier = NuSVC(nu=0.5, kernel='linear')
        classifier = classifier.fit(train_data.T, train_labels)
        predicted_labels = classifier.predict(image_data_shared[subject].T)
        accuracy[subject] = sum(predicted_labels == test_labels)/float(len(predicted_labels))

        classifier_dpsrm = NuSVC(nu=0.5, kernel='linear')
        classifier_dpsrm = classifier_dpsrm.fit(train_data_dpsrm.T, train_labels)
        predicted_labels_dpsrm = classifier_dpsrm.predict(image_data_shared_dpsrm[subject].T)
        accuracy_dpsrm[subject] = sum(predicted_labels_dpsrm == test_labels)/float(len(predicted_labels_dpsrm))

        classifier_dpsrm_orthow = NuSVC(nu=0.5, kernel='linear')
        classifier_dpsrm_orthow = classifier_dpsrm_orthow.fit(train_data_dpsrm_orthow.T, train_labels)
        predicted_labels_dpsrm_orthow = classifier_dpsrm_orthow.predict(image_data_shared_dpsrm_orthow[subject].T)
        accuracy_dpsrm_orthow[subject] = sum(predicted_labels_dpsrm_orthow == test_labels)/float(len(predicted_labels_dpsrm_orthow))


    print("SRM: The average accuracy among all subjects is {0:f} +/- {1:f}".format(np.mean(accuracy), np.std(accuracy)))
    print("MN-SRM: The average accuracy among all subjects is {0:f} +/- {1:f}".format(np.mean(accuracy_dpsrm), np.std(accuracy_dpsrm)))
    print("MN-SRM with orthogonal W: The average accuracy among all subjects is {0:f} +/- {1:f}".format(np.mean(accuracy_dpsrm_orthow), np.std(accuracy_dpsrm_orthow)))
