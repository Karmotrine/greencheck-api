import sys

from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler

sys.path.append(r'lib')
sys.path.append(r'utilities')

import os
import cv2
import numpy as np
from enum import Enum
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from utilities.features import *
np.set_printoptions(threshold=10)
"""
INITIAL PAThS
"""
MODEL_PATH = r'dataset\model\models'
SCALER_PATH = r'dataset\model\scalers'
FEATURE_PATH = r'dataset\model\features'
SEGMENTED_PATH = r'dataset\segmented'
AUGMENTED_PATH = r'dataset\augmented'
DATA_PATH = r'dataset\model'
CAPTURED_PATH = r'dataset\captured'
NEW_PATH = r'dataset\training'
TRAINING_PATH = r'dataset\finalized\training'
VALIDATION_PATH = r'dataset\finalized\validation'
TESTING_PATH = r'dataset\finalized\testing'
UNLABLED_PATH = r'dataset\plants'
PHILRICE_PATH = r'dataset\philrice'
DATASET_PATH = r'dataset\mix and match\ONNO'
SEG_PATH = r'dataset\model'
LOGS_PATH = r'reports'

"""
GENERAL CONSTANTS
"""
class ModelType(Enum):
    BaseModel       = 0
    ParticleSwarm   = 1
    AntColony       = 2
    ArtificialBee   = 3

class Disease(Enum):
    blb     =   0
    hlt     =   1
    rb     =   2
    sb     =   3

label_encoder = LabelEncoder()

DISEASES = ['blb', 'hlt', 'rb', 'sb']
CLASSIFIER = SVC(C=10, kernel='rbf', probability=True)
CORES = os.cpu_count() // 2
CORES = CORES if CORES // 2 >= os.cpu_count() else CORES + 1
PARAM_GRID = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto'] + [0.1, 1],
    # 'coef0': [0.0, 2.0],
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovr', 'ovo'],
    # 'shrinking': [True, False],
    'probability': [True, False],  # Add this line to control the randomness of the underlying implementation
    # 'random_state': [None, 0, 42]  # Add this line to control the seed of the random number generator
}
FOLDS = 4 # Amount of folds, KFOLD is automatically applied if FOLDS > 1
SHUFFLE = True # False to ensure replicability over all models

R_STATE = None # Select Random State to ensure replicablity
TEST_SIZE = 0.2 #  Percentage of test size
VAL_SIZE = 0.25
"""
OBJECTIVE FUNCTION
"""

def fitness_cv(features, labels, subset):
    assert FOLDS > 1, "Folds must be greater than 1"
    selected_features = features[:, subset]

    scores = []

    kfold = KFold(n_splits=FOLDS, shuffle=SHUFFLE, random_state=R_STATE)
    for train_index, test_index in kfold.split(selected_features):
        scaler = StandardScaler()
        svm = SVC(C=10, kernel='rbf', probability=True)
        X_train, X_val = selected_features[train_index], selected_features[test_index]
        Y_train, Y_val = labels[train_index], labels[test_index]
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        svm.fit(X_train, Y_train)
        Y_pred = svm.predict(X_val)
        accuracy = accuracy_score(Y_val, Y_pred)
        scores.append(accuracy)

    accuracy = np.array(scores).mean()
    return accuracy

def fitness(features, labels, subset):
    scaler = StandardScaler()
    X_train, X_val, Y_train, Y_val = train_test_split(features[:, subset], labels, test_size=VAL_SIZE, random_state=R_STATE)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    svm = SVC(C=10, kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)
    Y_pred = svm.predict(X_val)
    accuracy = accuracy_score(Y_val, Y_pred)

    return accuracy

def fitness_pso(features, labels, subset):

    if np.sum(subset) == 0:
        score = 1.0 
    selected_features = features[:, subset]

    if FOLDS > 1:
        kfold = KFold(n_splits=FOLDS, shuffle=SHUFFLE, random_state=R_STATE)
        scores = cross_val_score(CLASSIFIER, selected_features, labels, cv=kfold)
        accuracy = scores.mean()
    else:
        X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=TEST_SIZE, random_state=R_STATE)
        CLASSIFIER.fit(X_train, y_train)
        y_pred = CLASSIFIER.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    score = 1.0 - accuracy
    return score, selected_features

"""
FOR PRE-PROCESSING
"""
WIDTH = HEIGHT = 100
LTHRESHOLD = 128
DENOISE_KERNEL = (3, 3)
DENOISE_SIGMA = 0

LB = 191
UB = 251

SHARPEN_KERNEL = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], dtype=np.float32)

AUGMENTATIONS = [
    ('H_FLIP', lambda img: cv2.flip(img, 1)),
    ('V_FLIP', lambda img: cv2.flip(img, 0)),
    ('ROT90C', lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
    ('ROT180', lambda img: cv2.rotate(img, cv2.ROTATE_180)),
    ('ROT90O', lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ('CONINC', lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0)),
    ('CONDEC', lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=0)),
    ('BR_INC', lambda img: cv2.convertScaleAbs(img, alpha=1, beta=50)),
    ('BR_DEC', lambda img: cv2.convertScaleAbs(img, alpha=1, beta=-50)),
    ('IMBLUR', lambda img: cv2.GaussianBlur(img, (5, 5), 0)),  # Apply Gaussian blur
    ('SHARPN', lambda img: cv2.filter2D(img, -1, SHARPEN_KERNEL)),  # Apply sharpening filter
]
"""
FOR FEATURE EXTRACTION
"""
# HOG (SHAPE) FEATURES
PPC = (8, 8)
CPB = (1, 1)
ORIENT = 8

# LBP (TEXTURE) FEATURES
RADIUS = 1
POINTS = 8 * RADIUS

# GLCM (TEXTURE) FEATURES
DISTANCE = 1
ANGLES = 0
LEVELS = 256

# COLOR COHERANCE FEATURES
N_BINS = 8

FEATURES = {
    'SHP-HOG' : lambda image: getHOGFeatures(image, ORIENT, PPC, CPB), 
    'TXR-GLCM': lambda image: getGLCMFeatures(image, DISTANCE, ANGLES, LEVELS),
    'TXR-LBP' : lambda image: getLBPFeatures(image, RADIUS, POINTS),
    'COL-HSV' : lambda image: getHSVFeatures(image),
    'COL-LAB' : lambda image: getLABFeatures(image),
    'COL-RGB' : lambda image: getRGBFeatures(image),
    'COL-STAT' : lambda image: getStatFeatures(image),
    'COL-COLHIST': lambda image: getColHistFeatures(image),
    'COL-CCV': lambda image: getCCVFeatures(image, N_BINS)
}

GROUPED_FEATURES = {}
for feature, _ in FEATURES.items():
    prefix = feature.split('-')[0]  # Extract the prefix
    if prefix not in GROUPED_FEATURES:
        GROUPED_FEATURES[prefix] = []
    GROUPED_FEATURES[prefix].append(feature)