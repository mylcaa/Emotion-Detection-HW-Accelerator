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

# Emotion list
emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

# 68 point face model
PREDICTOR_PATH = "/home/catic/Documents/project/AdaBoost_DLib/shape_predictor_68_face_landmarks"
# face detector
detector = dlib.get_frontal_face_detector()
# 68 points predictor
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Define landmark pairs
landmark_pairs = [(37, 41), (38, 40), (43, 47), (44, 46),
                  (21, 22), (19, 28), (20, 27), (23, 27),
                  (24, 28), (48, 29), (31, 29), (35, 29),
                  (54, 29), (60, 64), (61, 67), (51, 57),
                  (62, 66), (63, 65), (18, 29), (25, 29)
                  ] 

#Definition of a function to later use in other files
def landmark_detector_dlib(image) :
    
    #image = imutils.resize(image, width=500)q
    #image = cv2.resize(image, [500, 500])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    faces = detector(gray, 1)
    # go through the face bounding boxes
    if len(faces) < 1:
        #In case no case selected, print "error" values
        #u slucaju da nema detektovanog lica vrati error
        #vrati landmarkse
        return np.zeros((68, 2))
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

#underneath is the code for testing without dlibs recognition of a face
'''
def landmark_detector_dlib(image) :
    
    #image = imutils.resize(image, width=500)
    image = cv2.resize(image, [500, 500])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    faces = detector(gray, 1)
    #new_faces = detect_bounding_box(gray)
    # go through the face bounding boxes
    if len(faces) < 1:
        #In case no case selected, print "error" values
        #u slucaju da nema detektovanog lica vrati error
        #vrati landmarkse
        return np.zeros((68, 2))

    #if len(new_faces) < 1:
        #In case no case selected, print "error" values
        #u slucaju da nema detektovanog lica vrati error
        #vrati landmarkse
    #    return np.zeros((68, 2))
    else:
        #for face in faces:        
            # apply the shape predictor to the face ROI
        #    shape = predictor(gray, dlib.rectangle(new_faces[0], new_faces[1], new_faces[2], new_faces[3]))
        shape = predictor(gray, faces)

        landmarks = np.zeros((68, 2))
        #print(result)
        for n in range(17, 68): # NIJE POTREBAN OKVIR FACE!
                x = shape.part(n).x
                y = shape.part(n).y
                #print("Landmark ID: ", n, "X coordinate: ", x, "Y coordinate: ", y)
                landmarks[n] = x, y
    
        return landmarks

'''



def distance_calculator_test(frame, landmark_pairs): 
    distances = [np.sqrt((frame[p1][0] - frame[p2][0])**2 +
                        (frame[p1][1] - frame[p2][1])**2)
                for p1, p2 in landmark_pairs]
    return distances

if __name__ == '__main__': 
    
    pkl_file = open('/home/catic/Documents/project/AdaBoost_DLib/predicted_model/model_new.pkl', 'rb')
    model_trained = pickle.load(pkl_file)    
    pkl_file.close()
    
    '''testing_data = []
    image_test = cv2.imread("C:/Users/User/Documents/ml_projekat/data/happyjoey.jpg")
    image_test_landmark = landmark_detector_dlib(image_test)
    print(len(image_test_landmark))
    image_test_distance = distance_calculator_test(image_test_landmark, landmark_pairs)
    print(len(image_test_distance))
    testing_data.append(image_test_distance)
    image_test_distance_array = np.array(testing_data)
    #image_test_distance_array = image_test_distance_array.reshape(1,-1)
    predicted_label = model_trained.predict(image_test_distance_array)
    predicted_label = predicted_label[0]
    print("Real Label: Surprised", "Predicted Label: ", predicted_label)'''
    
    video_capture = cv2.VideoCapture(0) 
    
    while True:
        testing_data = []
        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break  # terminate the loop if the frame is not read successfull
        
        #video_frame = cv2.imread("/home/catic/Documents/project/happy.jpg")
        video_frame_cpy = video_frame
        
        video_frame_landmarks = landmark_detector_dlib(video_frame)

        if (np.all(video_frame_landmarks == 0)):
            #cv2.imshow("AdaBoost Prediction ",video_frame_cpy)   
            print("Error")
        else:     
            video_frame_distance = distance_calculator_test(video_frame_landmarks, landmark_pairs)
            testing_data.append(video_frame_distance)
            video_frame_distance_array = np.array(testing_data) 
            video_frame_distance_array = video_frame_distance_array.reshape(1,-1)            
            result_prediction = model_trained.predict(video_frame_distance_array)
            #print("Emotion Prediction: ", emotions[result_prediction[0]])

            cv2.putText(video_frame_cpy ,emotions[result_prediction.item()], (30, 30), fontFace= cv2.FONT_HERSHEY_PLAIN, fontScale= 1.5, color =(0, 255, 0))
            cv2.imshow("AdaBoost Prediction ",video_frame_cpy)

        if cv2.waitKey(2) & 0xFF == ord("q"):
            break
 
    video_capture.release()
    cv2.destroyAllWindows()
