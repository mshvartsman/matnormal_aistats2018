## heavily based on SRM brainiak example
import scipy.io
import os
from sklearn.svm import NuSVC
import numpy as np
import models
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)

input_path = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider'
outfile_template = '/mnt/jukebox/pniintel/cohen/ms44/nips2017_data/raider/results_%i.csv'

def run_experiment(modelname):
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

    model = getattr(models, modelname)

    transformed_data = model(movie_data, image_data)

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
        TRs = transformed_data[0].shape[1]
        train_data = np.zeros((transformed_data[0].shape[0], len(train_labels)))

        for train_subject in range(len(train_subjects)):
            start_index = train_subject*TRs
            end_index = start_index+TRs
            train_data[:, start_index:end_index] = transformed_data[train_subjects[train_subject]]

        # Train a Nu-SVM classifier using scikit learn
        classifier = NuSVC(nu=0.5, kernel='linear')
        classifier = classifier.fit(train_data.T, train_labels)
        predicted_labels = classifier.predict(transformed_data[subject].T)
        accuracy[subject] = sum(predicted_labels == test_labels)/float(len(predicted_labels))
        train_acc[subject] = sum(classifier.predict(train_data.T) == train_labels)/float(len(train_labels))

    return {'mean_acc': np.mean(accuracy), 'training_mean_acc': np.mean(train_acc), "std_acc": np.std(accuracy), "std_train_acc" : np.std(train_acc)}

if __name__ == "__main__":

    try:
        myID = int(os.environ["SLURM_ARRAY_TASK_ID"])
        totalIDs = int(os.environ["SLURM_ARRAY_TASK_MAX"])
    except KeyError:
        myID = 1
        totalIDs = 1

    print("Job %s of %s reporting in!" % (myID, totalIDs))

    pointsPerId = len(models.models) / totalIDs
    start = (myID-1)*pointsPerId
    end = len(models.models) if myID == totalIDs else (myID)*pointsPerId
    print("Doing Params %s to %s (inclusive)" % (start, end-1))
    my_models = models.models[int(start):int(end)]

    res = pd.DataFrame([run_experiment(model) for model in my_models])

    fname = outfile_template % myID

    res.to_csv(fname)
    print("Done!")
