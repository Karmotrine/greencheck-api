import cv2
import joblib
from const import *

def segment(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    v = cv2.equalizeHist(v)
    v = cv2.convertScaleAbs(v, alpha=1.25)
    _, v = cv2.threshold(v, LB, UB, cv2.THRESH_BINARY)
    
    # Thresholding based segmentation
    mask = cv2.bitwise_or(s, v)
    _, mask = cv2.threshold(mask, LB, UB, cv2.THRESH_BINARY)
    image = cv2.bitwise_and(image, image, mask=mask)

    return image

def loadImages(dataset_path=TRAINING_PATH):
    features = []
    labels = []
    # Loop through the class folders
    for class_folder in os.listdir(dataset_path):
        class_label = class_folder
        class_path = os.path.join(dataset_path, class_folder)

        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path) 
            seg_image = segment(image)
            seg_image = cv2.resize(seg_image, (WIDTH, HEIGHT))
            
            feature = extractFeatures(seg_image)
            features.append(feature)
            labels.append(class_label)

            for _, augment in AUGMENTATIONS:
                aug_image = augment(image)
                aug_image = segment(aug_image)
                aug_image = cv2.resize(aug_image, (WIDTH, HEIGHT))

                feature = extractFeatures(aug_image)
                features.append(feature)
                labels.append(class_label)
    
    features = np.array(features)
    labels = np.array(labels)
    np.save(f"{DATA_PATH}/features.npy", features)
    np.save(f"{DATA_PATH}/labels.npy", labels)
    return features, labels

def preLoadImages():
    features = np.load(f"{DATA_PATH}/features.npy")
    labels = np.load(f"{DATA_PATH}/labels.npy")

    return features, labels

def loadUnseenImages():
    features = []
    labels = []
    # Loop through the class folders
    for class_folder in os.listdir(TESTING_PATH):
        class_label = class_folder
        class_path = os.path.join(TESTING_PATH, class_folder)

        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path) 
            seg_image = segment(image)
            seg_image = cv2.resize(seg_image, (WIDTH, HEIGHT))
            
            feature = extractFeatures(seg_image)
            features.append(feature)
            labels.append(class_label)
    
    features = np.array(features)
    labels = np.array(labels)

    return features, labels

def extractFeatures(image):
    features = []
    for _, feature_func in FEATURES.items():
        features.extend(feature_func(image))
    return np.array(features)
