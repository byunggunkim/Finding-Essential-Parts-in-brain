import tensorflow as tf
import os
import numpy as np
import pickle
import collections
from sklearn.model_selection import train_test_split

# LOSO dataset
def make_train_testset(dataset_path, test_site):
    trainset_list =  os.listdir(dataset_path)
    trainset_data = list()
    trainset_label = list()
    testset_data = list()
    testset_label = list()

    for i in trainset_list:
        # open data from site
        if i[0] == 't' and i.find(test_site) == -1:
            trainset_path = os.path.join(dataset_path, i)
            with open(trainset_path,'rb') as f:
                # data list of (t_series, label)
                train_data = pickle.load(f)
                train_label = pickle.load(f)
                trainset_data.extend(train_data)
                trainset_label.extend(train_label)
        elif i[0] == 't' and i.find(test_site) != -1:
            testset_path = os.path.join(dataset_path, i)
            with open(testset_path,'rb') as f:
                test_data = pickle.load(f)
                test_label = pickle.load(f)
                testset_data.extend(test_data)
                testset_label.extend(test_label)

    global min_class_number
    _, counts = np.unique(trainset_label, return_counts=True)
    min_class_number = np.min(counts)

    trainset = tf.data.Dataset.from_tensor_slices((tf.ragged.constant(trainset_data), tf.constant(trainset_label)))
    validset = (tf.ragged.constant(test_data), tf.one_hot(testset_label, depth=2))
    valid_eval = tf.data.Dataset.from_tensor_slices((tf.ragged.constant(testset_data), tf.constant(testset_label)))

    # shuffle trainset and testset
    trainset = trainset.shuffle(buffer_size = len(trainset_data))
    valid_eval = valid_eval.shuffle(buffer_size = len(testset_data)).batch(32).map(lambda x, y: (x, tf.one_hot(y, depth=2)))
    
    global trainset_len
    global testset_len
    
    trainset_len = len(trainset_data)
    testset_len = len(testset_data)

    return trainset, validset, valid_eval 


def count(counts, batch):
    features, labels = batch
    class_1 = labels == 1
    class_1 = tf.cast(class_1, tf.int32)

    class_0 = labels == 0
    class_0 = tf.cast(class_0, tf.int32)

    counts['class_0'] += tf.reduce_sum(class_0)
    counts['class_1'] += tf.reduce_sum(class_1)

    return counts


def balanced_resampling(dataset, batch_size):
    negative_ds = (
        dataset
        .filter(lambda features, label: label==0)
        .repeat())
    
    positive_ds = (
        dataset
        .filter(lambda features, label: label==1)
        .repeat())
    
    choice_dataset = tf.data.Dataset.range(2).repeat(int(min_class_number))
    balanced_batch = tf.data.experimental.choose_from_datasets([negative_ds, positive_ds],choice_dataset).batch(batch_size).map(lambda x, y: (x, tf.one_hot(y, depth=2)))
    
    return balanced_batch


if __name__ == "__main__":
    dataset_path = './preprocessed/all'
    trainset, validset, testset = make_train_testset(dataset_path, 'nyu')
    balanced_batch = balanced_resampling(trainset, 32)
