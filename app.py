from flask import Flask, render_template,request
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
model = keras.models.load_model('best_model.keras')

@app.route('/')
def index():
    return render_template("index.html", name="RGK")


@app.route('/prediction', methods=["POST"])
def prediction():
    img = request.files['img']
    img.save('img.jpg')
    img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32')/ 255
    img = img.reshape(-1, 150, 150, 1)
    # img = image.load_img("img.jpg", target_size=(150,150))
    # x=image.img_to_array(img) / 255
    # resized_img_np = np.expand_dims(x,axis=0)
    prediction = model.predict(img)

    if prediction > 0.5:
        pred = "la personne est normale"
    else:
        pred = "la personne a une pneumonie"    

    return render_template("prediction.html", data=pred)


if __name__ =="__main__":
    app.run(debug=True)