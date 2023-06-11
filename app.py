from flask import Flask, request,send_file
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import pickle as pk
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
with open('cbir_nn.pkl', 'rb') as f:
    nn = pk.load(f)
model=load_model('cbir_vgg16.h5')
# Load xtrain from a file
xtrain = np.load('xtrain.npy')

@app.route('/')
def index():
    return send_file(r'C:\Users\Saimum Adil Khan\OneDrive\Desktop\Flask\CBIR similar image search\templates\home.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_image = request.files['image'].read()
    input_image = np.frombuffer(input_image, np.uint8)
    input_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)
    input_image_resized = cv2.resize(input_image, (224, 224), interpolation=cv2.INTER_LINEAR)
    input_image_rgb = cv2.cvtColor(input_image_resized, cv2.COLOR_BGR2RGB)

    inp_img_preprocess = preprocess_input(input_image_rgb.reshape(1, 224, 224, 3))
    inp_img_feature = model.predict(inp_img_preprocess)
    inp_img_flat = inp_img_feature.reshape(1, -1)
    distances, indices = nn.kneighbors(inp_img_flat)

    # Get the similar images from xtrain
    similar_images = xtrain[indices]

    # Convert the similar images to base64-encoded strings
    similar_images_base64 = []
    for img in similar_images[0]:
        img_pil = Image.fromarray(img.astype('uint8'))
        buffer = BytesIO()
        img_pil.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        similar_images_base64.append(img_base64)

    return {'similar_images': similar_images_base64}

if __name__ == '__main__':
    app.run(debug=True)
