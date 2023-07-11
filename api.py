from flask import Flask, request
from PIL import Image
import io
import json
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

app = Flask(__name__)

# Constants
MODEL_PATH = 'fashion_model.h5'
LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the pre-trained Keras model
model = load_model(MODEL_PATH, compile=False)

@app.route('/', methods=['GET'])
def hello():
    return 'Fashion model'

@app.route('/getLabels', methods=['GET'])
def get_labels():
    return json.dumps(LABELS)

def process_image(image):
    """Resize, convert to grayscale, normalize and expand dimensions of the image."""
    image = image.resize((28, 28))  # Resize the image
    image = image.convert('L')  # Convert the image to grayscale
    img_array = img_to_array(image)  # Convert the PIL image to a numpy array
    img_array /= 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to represent a single 'batch' of images
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file.stream)  # Convert the file to an image
    image_array = process_image(image)  # Process the image
    predictions = model.predict(image_array)  # Run the image through the model and get the predictions
    y = np.argmax(predictions, axis=1)[0]  # Find the class with the highest probability
    result = {'label': LABELS[y], 'label_code': int(y), 'confidence': int(predictions.max() * 100)}
    return json.dumps(result)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5007)
