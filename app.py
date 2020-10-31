# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:46:04 2020

@author: sarath
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.transform import resize
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from werkzeug.utils import secure_filename
import cv2
import os

app = Flask(__name__)

model=load_model('dogcat.h5')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    # frame=cv2.imread(img)
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    if(np.max(x)>1):
            x=x/255.0

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')
    # img=resize(frame,(64,64))
    # img=np.expand_dims(img,axis=0)
    # if(np.max(img)>1):
    #     img=img/255.0

    preds = model.predict_classes(x)[0]
    return preds


@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        ls=["Cat","Dog"]
        result = ls[preds[0]]
        return result
    return None




if __name__ == '__main__':
      app.run(debug=True)
