"""
Python module which contains all the necessary functions for handling with the models.
"""
import pickle
import cv2    
import numpy as np
import joblib

from keras.preprocessing import image 
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

dog_names = pickle.load(open('dognames.pkl', 'rb'))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def extract_Resnet50(tensor):
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def dog_detector(img_path, model):
    img = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(model.predict(img))
        
    return ((prediction <= 268) & (prediction >= 151)) 


def face_detector(img_path, face_cascade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def predict_breed(img_path, model):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
