import json
import plotly
import pandas as pd
import numpy as np
import cv2
import pickle
import os

from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50

import model_utils

# Defining global variables
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')
ResNet50_model_dogdetector = ResNet50(weights='imagenet')
ResNet50_model_breeddetector = load_model(f"../saved_models/best_model")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
      
    # render web page with plotly graphs
    return render_template('master.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def predict():
    
    # Uploading and saving the image (from: https://flask.palletsprojects.com/en/2.0.x/patterns/fileuploads/)
    if request.method == 'POST':
        
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template('master.html')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template('master.html')            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'], filename)
            print(file_path)
            file.save(file_path)
    

    # Classifying the image
    image_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dog_names_pred = model_utils.predict_breed(image_path, ResNet50_model_breeddetector)

    output_str = ""
    if model_utils.dog_detector(image_path, ResNet50_model_dogdetector):
        output_str = f"In this photo is a dog and it looks like it is a {dog_names_pred.replace('_', ' ')}."
    elif model_utils.face_detector(image_path, face_cascade):
        output_str = f"In this photo is a human, which most resembles a {dog_names_pred.replace('_', ' ')}."
    else:
        output_str = "There was neither a dog nor a human detected in this image."


    return render_template('master.html', image_path=image_path, output_str=output_str)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()