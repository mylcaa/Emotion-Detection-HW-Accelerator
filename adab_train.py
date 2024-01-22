import cv2
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dlib
import os
import glob
import random
import matplotlib.pyplot as plt
import pickle

#TODO: smanjiti velicinu matrice landmarks u landmark_detector_dlib sa 68 na 68-16

# 68 point face model
PREDICTOR_PATH = "/home/catic/Documents/project/AdaBoost_DLib/shape_predictor_68_face_landmarks"
# face detector
detector = dlib.get_frontal_face_detector()
# 68 points predictor
predictor = dlib.shape_predictor(PREDICTOR_PATH)

emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def get_files(emotion): #koristi se za dobijanje skupa test i trening podataka
    images = glob.glob("/home/catic/Documents/project/other/DataSetCK+/CK+48/%s//*" %emotion) #glob biblioteka->pronalazi 
    #sve datoteke u odredjenom direktorijuumu koji sadzi slike za odredjenu emociju
    #datoteke se smijestaju u listu image
    random.shuffle(images) #pronadjene slike se mijesaju da ide podjednako na test i trening
    training_set = images[:int(len(images)*0.8)]   #get 80% of image files to be trained
    #trening set 80% dataseta
    testing_set = images[-int(len(images)*0.2):]   #get 20% of image files to be tested
    return training_set, testing_set
    #20% dataseta

#Definition of a function to later use in other files
def landmark_detector_dlib(image) :
    
    #image = imutils.resize(image, width=500)q
    image = cv2.resize(image, [250, 250])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    faces = detector(gray, 1)
    # go through the face bounding boxes
    if len(faces) < 1:
        #In case no case selected, print "error" values
        #u slucaju da nema detektovanog lica vrati error
        return np.zeros((68, 2))
        #vrati landmarkse
        #return landmarks
    else:
        for face in faces:        
            # apply the shape predictor to the face ROI
            shape = predictor(gray, face)

        landmarks = np.zeros((68, 2))
        #print(result)
        for n in range(17, 68): # NIJE POTREBAN OKVIR FACE!
                x = shape.part(n).x
                y = shape.part(n).y
                #print("Landmark ID: ", n, "X coordinate: ", x, "Y coordinate: ", y)
                landmarks[n] = x, y
    
        return landmarks

def make_sets(): #generise skup podataka za treniranje i testiranje
    training_data = []
    training_label = []
    testing_data = []
    testing_label = [] #liste za ove skupove podataka
    for emotion in emotions: #iteracija kroz listu emocija
        training_set, testing_set = get_files(emotion) #dodavanje trening i test setova
        #za odredjenu emociju
        #add data to training and testing dataset, and generate labels 0-4
        print("EMOTION: ", emotion)
        #print("Length of training set: ", len(training_set), "Length of testing_set", len(testing_set))
        for item in training_set:
            #print("Emotion: ", emotion, "Item_index:", training_set.index(item))
            #read image
            img = cv2.imread(item)
            #convert to grayscale
            #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #clahe_img = clahe.apply(gray_img) #poboljsanje kontrasta na sivoj slici
            landmarks_vec = landmark_detector_dlib(img) #stavaljaju se landmarks na sliku

            if np.all(landmarks_vec <= 0):
                pass
            else:
                #landmarks se stavljaju na trening set
                training_data.append(landmarks_vec)
                training_label.append(emotions.index(emotion)) #indeks emocije

        #analogno za test set
        for item in testing_set:
            img = cv2.imread(item)
            #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #clahe_img = clahe.apply(gray_img)
            landmarks_vec = landmark_detector_dlib(img)
            if np.all(landmarks_vec <= 0):
                pass
            else:
                testing_data.append(landmarks_vec)
                testing_label.append(emotions.index(emotion))
#vraca sve cetiri liste
    return training_data, training_label, testing_data, testing_label

# Define landmark pairs
landmark_pairs = [(37, 41), (38, 40), (43, 47), (44, 46),
                  (21, 22), (19, 28), (20, 27), (23, 27),
                  (24, 28), (48, 29), (31, 29), (35, 29),
                  (54, 29), (60, 64), (61, 67), (51, 57),
                  (62, 66), (63, 65), (18, 29), (25, 29)
                  ]  

# Calculate distances for each face
def distance_calculator(training_data, testing_data, landmark_pairs):
    feature_vectors_training = []
    for landmarks_for_face in training_data:
        distances = [np.sqrt((landmarks_for_face[p1][0] - landmarks_for_face[p2][0])**2 +
                            (landmarks_for_face[p1][1] - landmarks_for_face[p2][1])**2)
                    for p1, p2 in landmark_pairs]
        feature_vectors_training.append(distances)

    feature_vectors_testing = []
    for landmarks_for_face in testing_data:
        distances = [np.sqrt((landmarks_for_face[p1][0] - landmarks_for_face[p2][0])**2 +
                            (landmarks_for_face[p1][1] - landmarks_for_face[p2][1])**2)
                    for p1, p2 in landmark_pairs]
        feature_vectors_testing.append(distances)
    return feature_vectors_training, feature_vectors_testing

#Distance Calclutaro for testing later
def distance_calculator_test(frame, landmark_pairs): 
    distances = [np.sqrt((frame[p1][0] - frame[p2][0])**2 +
                        (frame[p1][1] - frame[p2][1])**2)
                for p1, p2 in landmark_pairs]
    return distances

if __name__ == '__main__':
    #slike_train, slike_test = get_files('fear') 
    for i in range(0,25): 
        training_data, training_label, testing_data, testing_label = make_sets()
        print("Length of Training Data: ", len(training_data))
        print("Length of training_label: ", len(training_label))
        print("Length of testing_data: ", len(testing_data))
        print("Length of testing_label: ", len(testing_label))
        print("Length of Training Data INDEX 0", len(training_data[0]))

        feature_vectors_training, feature_vectors_testing = distance_calculator(training_data, testing_data, landmark_pairs)
        print("Length of feature vectors training: ", len(feature_vectors_training), "Length of feature vector testing", len(feature_vectors_testing))
        print("Length of feature vectors training index 10: ", len(feature_vectors_training[10]), "Length of feature vector testing index 10", len(feature_vectors_testing[10]))
        
        # Convert to numpy array
        feature_array_training = np.array(feature_vectors_training)
        feature_array_testing = np.array(feature_vectors_testing)
        
        # Train the model
        max_accuracy = 0
    
        # Create a weak classifier (e.g., decision tree)
        base_classifier = DecisionTreeClassifier(max_depth=2)
        # Create AdaBoost classifier
        adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=100, random_state=42)
        adaboost_classifier.fit(feature_array_training, training_label)
        # Make predictions
        label_prediction = adaboost_classifier.predict(feature_array_testing)
        accuracy = accuracy_score(testing_label, label_prediction)
        print("Iteration: ", i, f" Accuracy: {accuracy}") 

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            adaboost_classifier_max = adaboost_classifier 

    
    print(f"Best Accuracy: {max_accuracy}") 
    i = 0
    for label in label_prediction:

        print("Predicted Label index: ", label, "Real Label index: ", testing_label[i])
        print("Predicted Label: ", emotions[label], "Real Label: ", emotions[testing_label[i]])
        i = i + 1

    #print('Best accuracy = ', max_accur*100, 'percent')
    #print(max_clf)
    
    testing_data = []
    image_test = cv2.imread("/home/catic/Documents/project/AdaBoost_DLib/vuk.jpg")
    image_test_landmark = landmark_detector_dlib(image_test)
    print(len(image_test_landmark))
    image_test_distance = distance_calculator_test(image_test_landmark, landmark_pairs)
    print(len(image_test_distance))
    testing_data.append(image_test_distance)
    image_test_distance_array = np.array(testing_data)
    #image_test_distance_array = image_test_distance_array.reshape(1,-1)
    predicted_label = adaboost_classifier_max.predict(image_test_distance_array)
    predicted_label = predicted_label[0]
    print("Real Label: Vuk", "Predicted Label: ", predicted_label)
    
    try:
        os.remove('/home/catic/Documents/project/AdaBoost_DLib/predicted_model/model_new.pkl')
    except OSError:
        pass
    output = open('/home/catic/Documents/project/AdaBoost_DLib/predicted_model/model_new.pkl', 'wb')
    pickle.dump(adaboost_classifier_max, output)
    output.close()
