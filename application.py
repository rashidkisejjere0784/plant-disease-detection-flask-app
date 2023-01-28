from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("plants.h5")

classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
       'Potato___Early_blight', 'Potato___Late_blight',
       'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
       'Tomato_Late_blight', 'Tomato_Leaf_Mold',
       'Tomato_Septoria_leaf_spot',
       'Tomato_Spider_mites_Two_spotted_spider_mite',
       'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
       'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods = ['GET', 'POST'])
def Predict():
    f = request.files['file'].read()
    npimg = np.fromstring(f, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))

    img = img / 255
    img = img.reshape((1,) + img.shape)
    output = np.argmax(model.predict(img))
    return render_template("index.html" , result = classes[output])


app.run(debug = True)