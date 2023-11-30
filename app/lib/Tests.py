import datetime
import itertools
import json
import time
import joblib
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import tqdm
from lib.Processing import segment
from lib.WrapperACO import WrapperACO
from lib.WrapperPSO import WrapperPSO

from const import *
def testAnalysis(self):
    FP = self.confusion.sum(axis=0) - np.diag(self.confusion)  
    FN = self.confusion.sum(axis=1) - np.diag(self.confusion)
    TP = np.diag(self.confusion)
    TN = self.confusion.sum() - (FP + FN + TP)

    TOTAL_FP = FP.sum()
    TOTAL_FN = FN.sum()
    TOTAL_TP = TP.sum()
    TOTAL_TN = TN.sum()

    # Overall accuracy
    PRECISION = TP/(TP+FP)
    RECALL = TP/(TP+FN)
    F1 = 2 * ((PRECISION * RECALL)/(PRECISION + RECALL))
    ACCURACY = (TP+TN)/(TP+FP+FN+TN)

    TOTAL_ACCURACY = (TOTAL_TP+TOTAL_TN)/(TOTAL_TP+TOTAL_FP+TOTAL_FN+TOTAL_TN)
    TOTAL_PRECISION = TOTAL_TP/(TOTAL_TP+TOTAL_FP)
    TOTAL_RECALL = TOTAL_TP/(TOTAL_TP+TOTAL_FN)
    TOTAL_F1 = 2 * ((TOTAL_PRECISION * TOTAL_RECALL)/(TOTAL_PRECISION + TOTAL_RECALL)) 

    stack = np.array((ACCURACY, PRECISION, RECALL, F1))
    labels = ['precision', 'recall', 'f1', 'accuracy']
    self.metrics = {
        'blb'   :   {v:c for c,v in zip(stack[:, 0], labels)},
        'hlt'   :   {v:c for c,v in zip(stack[:, 1], labels)},
        'rb'    :   {v:c for c,v in zip(stack[:, 2], labels)},
        'sb'    :   {v:c for c,v in zip(stack[:, 3], labels)},
        'total'     : {
            'precision' : TOTAL_PRECISION,
            'recall'    : TOTAL_RECALL,
            'f1'        : TOTAL_F1,
            'accuracy'  : TOTAL_ACCURACY,
        }
    }

def testPreProcess():
    test = 0
    disease = Disease.rb
    index = 1
    dataset = TRAINING_PATH
    if test:
        displayImages(
            blb=segment(cv2.imread(f'{dataset}/blb/{index}.jpg')),
            h=segment(cv2.imread(f'{dataset}/hlt/{index}.jpg')),
            rb=segment(cv2.imread(f'{dataset}/rb/{index}.jpg')),
            sb=segment(cv2.imread(f'{dataset}/sb/{index}.jpg')),
        )
    else:
        path = f'{dataset}/{disease.name}/{index}.jpg'
        # path = r'dataset\google\blb1.jpg'
        img = cv2.imread(path)
        
        # displayChannels(img, alpha=1.25, lower=200, mask=True)
        # displayImages(
        #     size=256,
        #     main=segment_leaf(img)
        # )

    stopWait()

def BaseModel(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create an MSVM model with an RBF kernel
    svm = CLASSIFIER

    # Train the model on the training data
    svm.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = svm.predict(X_test)

    # Generate a classification report
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')

    # Calculate the overall accuracy
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    # Print the classification report with class-wise accuracy
    print("Classification Report:")
    print(report)

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Features: {features.shape[0]}")
    
    print(label_encoder.classes_)
    return svm, overall_accuracy, None

def useBaseCV(features, labels, cv=5):
    # Convert class labels to numerical labels
    unique_labels = np.unique(labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numerical_labels = np.array([label_to_id[label] for label in labels])

    # Create a SimpleImputer to handle missing values (replace 'mean' with your preferred strategy)
    imputer = SimpleImputer(strategy='mean')

    # Apply imputation to your feature data
    X_imputed = imputer.fit_transform(features)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, numerical_labels, test_size=0.2, random_state=42)

    # Create an MSVM model with an RBF kernel
    svm = CLASSIFIER

    # Perform cross-validation and obtain scores
    cv_scores = cross_val_score(svm, X_imputed, numerical_labels, cv=cv)

    # Print cross-validation scores
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())

    # Fit the model on the entire training data
    svm.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = svm.predict(X_test)

    # Convert numerical labels back to original class labels
    predicted_class_labels = [unique_labels[label] for label in Y_pred]

    # Generate a classification report
    report = classification_report(Y_test, Y_pred, target_names=unique_labels, zero_division='warn')

    # Calculate the overall accuracy
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    # Print the classification report with class-wise accuracy
    print("Classification Report:")
    print(report)
    print(f"Features: {features.shape[0]}")

    return svm

def useGridSVC(features, labels, param_grid, cv=2):
    # Convert class labels to numerical labels
    
    numerical_labels = label_encoder.fit_transform(labels)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, numerical_labels, test_size=0.2, random_state=42)

    # Create an SVM classifier
    svm = SVC()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=10)

    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train)

    # Get the best parameters and estimator
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    # Use the best estimator to make predictions
    Y_pred = best_estimator.predict(X_test)

    # Convert numerical labels back to original class labels
    predicted_class_labels = label_encoder.inverse_transform(Y_pred)

    # Generate a classification report
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')

    # Calculate the overall accuracy
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    # Print the best parameters, overall accuracy, and classification report
    print("Best Parameters:", best_params)
    print("Best Estimators:", best_estimator)
    print("Overall Accuracy: {:.2f}%".format(overall_accuracy * 100))
    print("Classification Report:")
    print(report)

    return best_params, best_estimator

def createModel(features, labels, selectedFeatures=None):
    X_train, X_test = features
    Y_train, Y_test = labels

    X_train = X_train[:, selectedFeatures] if selectedFeatures is not None else X_train
    X_test = X_test[:, selectedFeatures] if selectedFeatures is not None else X_test

    svm = SVC(C=10, kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    Y_pred = svm.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')
    jsonreport = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, output_dict=True)
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    print("Classification Report:")
    print(report)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Features: {X_train.shape[1]}")

    try:
        with open(f'{LOGS_PATH}/ClassReports.json', 'r') as file:
            try:
                data = json.load(file)
            except json.decoder.JSONDecodeError:
                data = {'reports': []}
    except FileNotFoundError:
        with open(f'{LOGS_PATH}/ClassReports.json', 'w') as file:
            data = {'reports': []}
            json.dump(data, file, indent=4)

    with open(f'{LOGS_PATH}/ClassReports.json', 'w+') as file:
        data['reports'].append({f"Model-{len(data['reports'])+1}":jsonreport})
        json.dump(data, file, indent=4)
        
    return svm, overall_accuracy

def useCVTests(X, Y, selectedFeatures=None):
    selected_features = X[:, selected_features] if selectedFeatures is not None else X
    scores = []

    kfold = KFold(n_splits=FOLDS, shuffle=SHUFFLE, random_state=R_STATE)
    for train_index, test_index in kfold.split(selected_features):
        scaler = StandardScaler()
        svm = SVC(C=10, kernel='rbf', probability=True)
        X_train, X_test = selected_features[train_index], selected_features[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm.fit(X_train, Y_train)
        Y_pred = svm.predict(X_test)
        valid_score = accuracy_score(Y_test, Y_pred)
        scores.append(valid_score)

    valid_accuracy = np.array(scores).mean()
    print(f"Validated Overall Accuracy: {valid_accuracy}")

    scaler = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train[:, selectedFeatures] if selectedFeatures is not None else X_train
    X_test = X_test[:, selectedFeatures] if selectedFeatures is not None else X_test

    svm = SVC(C=10, kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    Y_pred = svm.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')
    jsonreport = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, output_dict=True)
    jsonreport['Validated Accuracy'] = valid_accuracy
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    print("Classification Report:")
    print(report)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Features: {X_train.shape[1]}")

    try:
        with open(f'{LOGS_PATH}/ClassReports.json', 'r') as file:
            try:
                data = json.load(file)
            except json.decoder.JSONDecodeError:
                data = {'reports': []}
    except FileNotFoundError:
        with open(f'{LOGS_PATH}/ClassReports.json', 'w') as file:
            data = {'reports': []}
            json.dump(data, file, indent=4)

    with open(f'{LOGS_PATH}/ClassReports.json', 'w+') as file:
        data['reports'].append({f"Model-{len(data['reports'])+1}":jsonreport})
        json.dump(data, file, indent=4)
        
    return svm, overall_accuracy

def useWrapperACO(aco: WrapperACO):
    print("Starting Ant Colony Optimization")
    solution, quality = aco.optimize()
    print("Optimization with Ant Colony Complete")
    print(f"Solution: {np.sort(solution)} with {100*quality:.2f}% accuracy")
    print(f"Ant Colony ", end=" ")
    return solution

def useWrapperPSO(features, labels, pso: WrapperPSO):
    print("Starting Particle Swarm Optimization")
    solution = pso.optimize()
    print("Optimization with Particle Swarm Complete")
    print(f"Solution: {np.sort(solution)}")
    print(f"Particle Swarm ", end=" ")
    model, accuracy = createModel(features, labels, solution)
    return model, accuracy, solution

def useSessionWrapperACO(aco: WrapperACO, fit, iterations, status, features=None, labels=None):
    match status:
        case 0: 
            solution, quality = aco.start_run(iterations)
            print(f"Solution: {np.sort(solution)} with {100*quality:.2f}% accuracy")
        case 1: 
            solution, quality = aco.continue_run(fit, iterations)
            print(f"Solution: {np.sort(solution)} with {100*quality:.2f}% accuracy")
        case 2: 
            assert features is not None and labels is not None
            solution, quality = aco.finish_run(fit, iterations)
            print(f"Solution: {np.sort(solution)} with {100*quality:.2f}% accuracy")
            model, accuracy = createModel(features, labels)
            saveModel(model, ModelType.AntColony, solution)
            
def exhaustiveFeatureTest():
    models = [ModelType.BaseModel]
    images = []
    mark = datetime.now().strftime('%Y%m%d-%H%M%S')
    # Generate Combinations of Features
    combinations = []
    for i in range(1, len(GROUPED_FEATURES) + 1):
        for subset in itertools.combinations(GROUPED_FEATURES, i):
            combination = []
            for item in subset:
                for feature_type in GROUPED_FEATURES[item]:
                    combination.append(feature_type)
            combinations.append(combination)

    print("Preparing Images")
    labels = []
    if os.path.exists(f"{DATA_PATH}/images.npy"):
        print("Images Loaded")
        images = np.load(f"{DATA_PATH}/images.npy")
        labels = np.load(f"{DATA_PATH}/labels.npy")
    else:
        path = TRAINING_PATH
        augment = True
        pre = False
        print("Processing Images")
        for class_folder in os.listdir(path):
            class_label = class_folder
            class_path = os.path.join(path, class_folder)
            print(f"Class Label: {class_label}")
            if not os.path.exists(class_path):
                continue 
            # Loop through the images in the class folder
            for image_file in tqdm(os.listdir(class_path)):
                image_path = os.path.join(class_path, image_file)
                image = cv2.imread(image_path) 
                seg_image = segment(image)
                seg_image = cv2.resize(seg_image, (WIDTH, HEIGHT))
                images.append(seg_image)
                labels.append(class_label)
                if augment:
                    for augmentation_name, augmentation_fn in AUGMENTATIONS:
                        aug_image = augmentation_fn(image if pre else seg_image)
                        if pre:
                            aug_image = segment(aug_image)
                            aug_image = cv2.resize(aug_image, (WIDTH, HEIGHT))
                        images.append(aug_image)
                        labels.append(class_label)

        Y = np.array(labels)
        Y = label_encoder.fit_transform(Y)
    print(f"Images: {Y.shape[0]}")
    save = False

    print("Starting Exhaustive Training")

    for combination in combinations:
        print("Combination: ", combination)
        X = []
        for image in images:
            img_feature = []
            for feature in combination:
                img_feature.extend(FEATURES[feature](image))
            img_feature = np.array(img_feature)
            X.append(img_feature)
        X = np.array(X)
        scaler = StandardScaler()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=R_STATE)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_split = X_train, X_test
        Y_split = Y_train, Y_test

        def fitness_function(subset): return fitness_cv(X_train, Y_train, subset) if FOLDS > 1 else fitness(X_train, Y_train, subset)
        subset = np.arange(0, X.shape[1])
        fit_accuracy = 0
        
        if combinations.index(combination) == len(combinations) - 1:
            save = True

        for model in models:
            start = time.time()
            
            try:
                with open(f'{LOGS_PATH}/{model.name}/{model.name}-{mark}.json', 'r') as file:
                    data = json.load(file)
            except FileNotFoundError:
                with open(f'{LOGS_PATH}/{model.name}/{model.name}-{mark}.json', 'w') as file:
                    data = {'tests': []}
                    json.dump(data, file, indent=4)
            if model is not ModelType.BaseModel:
                fit_accuracy = fitness_cv(X_train, Y_train, subset) if FOLDS > 1 else fitness(X_train, Y_train, subset)
                print(f"Initial: {subset.shape[0]}: {fit_accuracy}")
            match model:
                case ModelType.BaseModel:
                    classifier, accuracy = createModel(X_split, Y_split)
                case ModelType.AntColony:
                    aco = WrapperACO(fitness_function,
                                    X_train.shape[1], ants=5, iterations=5, rho=0.1, Q=.75, debug=1, accuracy=fit_accuracy, parrallel=True)
                    solution = useWrapperACO(aco)
                    classifier, accuracy, = createModel(X_split, Y_split, solution)
            if save:
                saveModel(classifier, scaler, model, subset)
                exec(open("predict.py").read())

            end = time.time()
            hours, remainder = divmod(int(end-start), 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            diseases = ['blb', 'hlt', 'rb', 'sb']
            testing = {
                'blb' : None,
                'hlt' : None,
                'rb' : None,
                'sb' : None,
            }
            validation = {
                'blb' : None,
                'hlt' : None,
                'rb' : None,
                'sb' : None,
            }

            for class_folder in os.listdir(TESTING_PATH):
                amount = 0
                correct = 0
                curclass = diseases.index(class_folder)
                for image_file in os.listdir(f"{TESTING_PATH}/{class_folder}"):
                    amount += 1
                    image_path = os.path.join(f"{TESTING_PATH}/{class_folder}", image_file)
                    test_image = cv2.imread(image_path) 
                    test_image = segment(test_image)
                    test_image = cv2.resize(test_image, (WIDTH, HEIGHT))
                    img_feature = []
                    for feature in combination:
                        img_feature.extend(FEATURES[feature](test_image))
                    img_feature = np.array(img_feature)
                    unseen = [img_feature]
                    unseen = scaler.transform(unseen)

                    if model is not ModelType.BaseModel:
                        unseen = unseen[:,solution]
                    prediction = classifier.predict(unseen)[0]
                    correct += 1 if prediction == curclass else 0
    
                testing[diseases[curclass]] = f"{(correct/amount)*100:.2f}%"


            for class_folder in os.listdir(VALIDATION_PATH):
                amount = 0
                correct = 0
                curclass = diseases.index(class_folder)
                for image_file in os.listdir(f"{VALIDATION_PATH}/{class_folder}"):
                    amount += 1
                    image_path = os.path.join(f"{VALIDATION_PATH}/{class_folder}", image_file)
                    test_image = cv2.imread(image_path) 
                    test_image = segment(test_image)
                    test_image = cv2.resize(test_image, (WIDTH, HEIGHT))
                    img_feature = []
                    for feature in combination:
                        img_feature.extend(FEATURES[feature](test_image))
                    img_feature = np.array(img_feature)
                    unseen = [img_feature]
                    unseen = scaler.transform(unseen)

                    if model is not ModelType.BaseModel:
                        unseen = unseen[:,subset]
                    prediction = classifier.predict(unseen)[0]
                    correct += 1 if prediction == curclass else 0
    
                validation[diseases[curclass]] = f"{(correct/amount)*100:.2f}%"

            log = {f"Test-{len(data['tests'])+1}": 
                {"Name": model.name, 
                    "Date": datetime.now().strftime('%Y/%m/%d %H:%M:%S'), 
                    "Elapsed": elapsed, 
                    'Image Size:' : f"{WIDTH}x{HEIGHT}", 
                    "Accuracy": f"{100*accuracy:.2f}%", 
                    "Saved": "True" if save else "False",
                    'Images': X.shape[0], 
                    "Features": 
                        {'Amount': subset.shape[0], 
                        'Feature': combination},  
                    "Testing (LB)": testing, 
                    "Validation (OL)": validation, 
                    'Additional': 'None' if model is ModelType.BaseModel else ({
                        'Ants': aco.ants,
                        'Iterations': aco.iterations,
                        'Rho': aco.rho,
                        'Q': aco.Q,
                        'Alpha': aco.alpha,
                        'Beta': aco.beta
                    } if model is ModelType.AntColony else 'None')}}
            
            with open(f'{LOGS_PATH}/{model.name}/{model.name}-{mark}.json', 'w+') as file:
                data['tests'].append(log)
                json.dump(data, file, indent=4)

def useCVTests(X, Y, selectedFeatures=None):
    selected_features = X[:, selected_features] if selectedFeatures is not None else X
    scores = []

    kfold = KFold(n_splits=FOLDS, shuffle=SHUFFLE, random_state=R_STATE)
    for train_index, test_index in kfold.split(selected_features):
        scaler = StandardScaler()
        svm = SVC(C=10, kernel='rbf', probability=True)
        X_train, X_test = selected_features[train_index], selected_features[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm.fit(X_train, Y_train)
        Y_pred = svm.predict(X_test)
        valid_score = accuracy_score(Y_test, Y_pred)
        scores.append(valid_score)

    valid_accuracy = np.array(scores).mean()
    print(f"Validated Overall Accuracy: {valid_accuracy}")

    scaler = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train[:, selectedFeatures] if selectedFeatures is not None else X_train
    X_test = X_test[:, selectedFeatures] if selectedFeatures is not None else X_test

    svm = SVC(C=10, kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    Y_pred = svm.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')
    jsonreport = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, output_dict=True)
    jsonreport['Validated Accuracy'] = valid_accuracy
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    print("Classification Report:")
    print(report)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Features: {X_train.shape[1]}")

    try:
        with open(f'{LOGS_PATH}/ClassReports.json', 'r') as file:
            try:
                data = json.load(file)
            except json.decoder.JSONDecodeError:
                data = {'reports': []}
    except FileNotFoundError:
        with open(f'{LOGS_PATH}/ClassReports.json', 'w') as file:
            data = {'reports': []}
            json.dump(data, file, indent=4)

    with open(f'{LOGS_PATH}/ClassReports.json', 'w+') as file:
        data['reports'].append({f"Model-{len(data['reports'])+1}":jsonreport})
        json.dump(data, file, indent=4)
        
    return svm, overall_accuracy

def displayImages(size=175, **imgs):
    for model, img in imgs.items():
        img = cv2.resize(img, (size, size))
        cv2.imshow(model, img)

def displayChannels(image, size=150, alpha=1, upper=255, lower=127, eq=True, mask=False):
    rgb = cv2.medianBlur(image, ksize=3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    l, a, B = cv2.split(lab)
    h, s, v = cv2.split(hsv)
    r, g, c = cv2.split(rgb)
    
    l = cv2.equalizeHist(l)
    a = cv2.equalizeHist(a)
    B = cv2.equalizeHist(B)
    h = cv2.equalizeHist(h)
    s = cv2.equalizeHist(s)
    v = cv2.equalizeHist(v)
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    c = cv2.equalizeHist(c)
    
    # a = 255-a
    # B = 255-B
    # l = 255-l

    l = cv2.convertScaleAbs(l, alpha=alpha)
    a = cv2.convertScaleAbs(a, alpha=alpha)
    B = cv2.convertScaleAbs(B, alpha=alpha)
    h = cv2.convertScaleAbs(h, alpha=alpha)
    s = cv2.convertScaleAbs(s, alpha=alpha)
    v = cv2.convertScaleAbs(v, alpha=alpha)
    r = cv2.convertScaleAbs(r, alpha=alpha)
    g = cv2.convertScaleAbs(g, alpha=2.5)
    c = cv2.convertScaleAbs(c, alpha=alpha)

    _, l = cv2.threshold(l, LB, upper, cv2.THRESH_BINARY)
    _, a = cv2.threshold(a, LB, upper, cv2.THRESH_BINARY)
    _, B = cv2.threshold(B, LB, upper, cv2.THRESH_BINARY)
    _, h = cv2.threshold(h, LB, upper, cv2.THRESH_BINARY)
    _, s = cv2.threshold(s, LB, upper, cv2.THRESH_BINARY)
    _, v = cv2.threshold(v, LB, upper, cv2.THRESH_BINARY)
    _, r = cv2.threshold(r, LB, upper, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, LB, upper, cv2.THRESH_BINARY)
    _, c = cv2.threshold(c, LB, upper, cv2.THRESH_BINARY)

    # Define the shape of the kernel (e.g., a square)
    kernel_shape = cv2.MORPH_RECT  # You can use cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, or cv2.MORPH_CROSS

    # Define the size of the kernel (width and height)
    kernel_size = (5, 5)  # Adjust the size as needed

    # Create the kernel
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    
    # Thresholding based segmentation
    # leaf = cv2.bitwise_xor(s, B)
    # leaf = cv2.bitwise_xor(h, s)
    # leaf = cv2.bitwise_not(leaf)
    # leaf = cv2.bitwise_xor(leaf, h)
    # leaf = cv2.bitwise_and(leaf, g)
    # leaf = cv2.bitwise_xor(leaf, v)
    # leaf = cv2.dilate(leaf, kernel, iterations=2)
    # leaf = cv2.bitwise_xor(leaf, s)
    leaf = cv2.bitwise_or(s, v)

    dise = cv2.bitwise_or(a, h)

    imask = cv2.bitwise_xor(dise, leaf)
    imask = leaf

    if mask:
        l = cv2.bitwise_and(image, image, mask=l)
        a = cv2.bitwise_and(image, image, mask=a)
        B = cv2.bitwise_and(image, image, mask=B)
        h = cv2.bitwise_and(image, image, mask=h)
        s = cv2.bitwise_and(image, image, mask=s)
        v = cv2.bitwise_and(image, image, mask=v)
        r = cv2.bitwise_and(image, image, mask=r)
        g = cv2.bitwise_and(image, image, mask=g)
        c = cv2.bitwise_and(image, image, mask=c)

    masked = cv2.bitwise_and(image, image, mask=imask)
    leaf = cv2.bitwise_and(image, image, mask=leaf)
    dise = cv2.bitwise_and(image, image, mask=dise)

    displayImages(
        Main=masked,
        l=l,
        a=a,
        B=B,
        y=h,
        s=s,
        v=v,
        r=r,
        g=g,
        c=c,
        p=dise,
        u=leaf,
    )   

def testseg(image):
    r, g, b = cv2.split(image)
    n, tg = cv2.threshold(g, 163, 255, cv2.THRESH_BINARY)
    g = cv2.bitwise_and(g, g, mask=tg)
    image = cv2.merge((r, g, b))
    displayImages(gs=g, rs=r,bs=b)

def stopWait():
    cv2.waitKey()
    cv2.destroyAllWindows()

def predictImage(image, model: ModelType):
    classifier = joblib.load(f"{MODEL_PATH}/{model.name}.joblib")
    scaler = joblib.load(f"{SCALER_PATH}/{model.name}.pkl")
    encoder = joblib.load(r'dataset\model\encoder.joblib')

    image = cv2.imread(image) 
    image = segment(image)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    X = [getFeatures(image)]
    X = scaler.transform(X)

    if model is not ModelType.BaseModelType:
        selected_features = np.load(f"{FEATURE_PATH}/{model.name}.npy")
        X = X[:,selected_features]
    
    prediction = classifier.predict_proba(X)
    print(encoder.classes_)

    return prediction

def saveModel(classifier, scaler, model, subset=None):
    print("ModelType Saving")
    joblib.dump(classifier, f"{MODEL_PATH}/{model.name}.joblib")
    joblib.dump(scaler, f"{SCALER_PATH}/{model.name}.pkl")
    if subset is not None:
        np.save(f"{FEATURE_PATH}/{model.name}.npy", subset)
    print("ModelType Saved")

def getFeatures(image):
    features = []
    for _, feature_func in FEATURES.items():
        features.extend(feature_func(image))
    return np.array(features)
