import numpy as np
import os
import pickle

ROOT = os.getcwd()+'/cifar-10-batches-py/'
def load_batch(filename):
    with open(filename, 'rb') as file:
        d = pickle.load(file, encoding='latin1')
        X = np.array(d['data']).reshape(10000,3,32,32).astype('float')
        y = np.array(d['labels'])
    return X, y

def load_CIFAR10(ROOT=ROOT):
    """Takes as input the ROOT directory containing CIFAR-10. returns X_train, y_train, X_test, y_test"""
    X_train = []
    y_train = []
    for i in range(1,6):
        print('loading batch '+str(i)+'...')
        X, y = load_batch(ROOT+'data_batch_'+str(i))
        X_train.append(X)
        y_train.append(y)
        del X, y
    print('Concatenating files')
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    print('Loading test data...')
    X_test, y_test = load_batch(ROOT+'test_batch')
    return X_train, y_train, X_test, y_test

def get_CIFAR10(train_size=0.96):
    """Takes as input the raw train and test data. Returns a test, validation and train set, in the form of
    a dictionary with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'"""
    X, y, X_test, y_test = load_CIFAR10()

    print('Shuffling training data...')
    mask = np.random.permutation(len(X))
    X_train, y_train, X_val, y_val = X[mask[:49000]], y[mask[:49000]], X[mask[49000:]], y[mask[49000:]]

    print('Subtracting feature means...')
    avg = np.mean(X_train, axis=0)
    X_train -= avg
    X_val -= avg
    X_test -= avg
    print('Finshed loading.')
    return {'X_train':X_train, 'X_val':X_val, 'X_test':X_test, 'y_train':y_train, 'y_val':y_val,'y_test':y_test}
