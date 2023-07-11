# Fashion Classification Flask App

This is a Flask application that uses a pre-trained Keras model to classify images of clothing items. The model can classify an image into one of ten categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Installation

To install the dependencies for this project, run the following command:

```
pip install flask pillow keras tensorflow wandb sklearn pandas
```
The pre-trained model file fashion_model.h5 needs to be present in the same directory as the script.

# Usage
## Flask Application
You can start the application by running the api.py script:
```
pip install flask pillow keras tensorflow wandb sklearn pandas

```
The application will start a web server that listens on port 5007. The following endpoints are available:
- 'GET /': Returns a simple greeting message.
- 'GET /getLabels': Returns the list of labels that the model can predict.
- 'POST /predict': Expects a file with key 'image'. Returns the predicted label, label code, and confidence score.

## Model Training
You can train the model by running the model.py script:
```
python model.py
```
This script uses the Fashion MNIST dataset, creates a CNN model, trains it, evaluates it, and saves it as fashion_model.h5. The script uses Weights & Biases (wandb) for experiment tracking and metrics logging. The output of the training process is saved in output.log.

## Log Output
The output.log file contains the log output from the training process. This includes information such as the structure of the model, the number of parameters in each layer, the training progress for each epoch, and the final evaluation metrics.

## Development
The process_image() function in api.py is used to prepare the image for prediction. It resizes the image to 28x28 pixels, converts it to grayscale, normalizes it, and expands its dimensions to represent a single 'batch' of images.

The load_and_preprocess_data(), create_model(), and train_model() functions in model.py are used to load and preprocess the Fashion MNIST data, create the CNN model, and train the model, respectively.

## Docker Deployment
The application can be containerized using Docker. The provided Dockerfile sets up an environment with Python 3.9.7, installs the necessary Python packages from the 'requirements.txt' file, and sets the default command to run the Flask application.

### Building the Docker Image
To build the Docker image, navigate to the directory containing the Dockerfile and run the following command:
```
docker build -t fashion_classification_app .
```
This command builds a Docker image and tags it as 'fashion_classification_app'.

### Running the Docker Container
After the image has been built, you can start a container with the following command:
```
docker run -p 5007:5007 fashion_classification_app
```
This command starts a Docker container and maps port 5007 in the container to port 5007 on your host machine, allowing you to access the Flask application at 'localhost:5007'.