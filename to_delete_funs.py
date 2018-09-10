import pandas as pd
import numpy as np
import h5py

def load_data(print_check=True):
    # Loading the data (cat/non-cat)
    train_dataset = h5py.File('./data/train_catvnoncat.h5', 'r')
    X_train = np.array(train_dataset["train_set_x"][:]) # train set features
    y_train_orig = np.array(train_dataset["train_set_y"][:]) # train set labels
    
    test_dataset = h5py.File('./data/test_catvnoncat.h5', "r")
    X_test = np.array(test_dataset["test_set_x"][:]) # test set features
    y_test_orig = np.array(test_dataset["test_set_y"][:]) # test set labels
    
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    y_train = y_train_orig.reshape((1, y_train_orig.shape[0]))
    y_test = y_test_orig.reshape((1, y_test_orig.shape[0]))

    m_train = y_train.shape[0]
    m_test = y_test.shape[0]
    num_px = X_train.shape[1]

    #X_train = X_train.T
    #y_train = y_train.T
    #X_test = X_test.T
    #y_test = y_test.T
    if print_check:
        print ("Number of training examples: m_train = " + str(m_train))
        print ("Number of testing examples: m_test = " + str(m_test))
        print ("Height/Width of each image: num_px = " + str(num_px))
        print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print ("train_set_x shape: " + str(X_train.shape))
        print ("train_set_y shape: " + str(y_train.shape))
        print ("test_set_x shape: " + str(X_test.shape))
        print ("test_set_y shape: " + str(y_test.shape))
        print()
        # print ("sanity check after reshaping: " + str(X_train[0:5,0]))
    
    return X_train, y_train, X_test, y_test


def flatten_data(X_train, X_test, print_check=True):
    X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
    X_test_flatten = X_test.reshape(X_test.shape[0], -1).T 
    
    print ("train_set_x_flatten shape: " + str(X_train_flatten.shape))
    #print ("train_set_y shape: " + str(y_train.shape))
    print ("test_set_x_flatten shape: " + str(X_test_flatten.shape))
    #print ("test_set_y shape: " + str(y_test.shape))
    print ("sanity check after reshaping: " + str(X_train_flatten[0:5,0]))

    return X_train_flatten, X_test_flatten