import warnings
import numpy as np
import pandas as pd
import wandb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

# Filter warnings
warnings.filterwarnings('ignore')

# Constants
NUM_CLASSES = 10
EPOCHS = 75
IMAGE_ROWS = 28
IMAGE_COLS = 28
BATCH_SIZE = 4096
IMAGE_SHAPE = (IMAGE_ROWS, IMAGE_COLS, 1)
CLASS_NAMES = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_and_preprocess_data():
    """Loads the fashion MNIST dataset and preprocesses it by normalizing and reshaping."""
    (train_images, y_train), (test_images, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = train_images / 255.0, test_images / 255.0

    x_train, x_validate, y_train, y_validate = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)

    # Reshape the images
    x_train = x_train.reshape(x_train.shape[0], *IMAGE_SHAPE)
    x_test = x_test.reshape(x_test.shape[0], *IMAGE_SHAPE)
    x_validate = x_validate.reshape(x_validate.shape[0], *IMAGE_SHAPE)

    return x_train, y_train, x_test, y_test, x_validate, y_validate

def create_model():
    """Creates a Sequential CNN model."""
    cnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=IMAGE_SHAPE),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return cnn_model

def train_model(model, x_train, y_train, x_test, y_test, config):
    """Compiles and trains the model, then evaluates it."""
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    model.compile(optimizer, config.loss_function, metrics=['acc'])

    _ = model.fit(x_train, y_train,
                  epochs=config.epochs,
                  batch_size=config.batch_size,
                  validation_data=(x_test, y_test),
                  callbacks=[WandbCallback()])

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy

def main():
    """Main function to run the script."""
    wandb.login()

    # Initialize wandb with your project name
    run = wandb.init(project='fashion_wandb_project',  # Can configure by our own
                     config={  # and include hyperparameters and metadata
                         "learning_rate": 0.001,
                         "epochs": EPOCHS,
                         "batch_size": BATCH_SIZE,
                         "loss_function": "sparse_categorical_crossentropy",
                         "architecture": "CNN",
                         "dataset": "FASHION"
                     })

    # Load and preprocess the data
    x_train, y_train, x_test, y_test, x_validate, y_validate = load_and_preprocess_data()

    # Initialize model
    tf.keras.backend.clear_session()
    model = create_model()
    model.summary()

    # Train and evaluate model
    loss, accuracy = train_model(model, x_train, y_train, x_test, y_test, run.config)        

    print('Test Error Rate: ', round((1 - accuracy) * 100, 2))
    
    # Save the model
    model.save("fashion_model.h5")

    # Log metrics
    wandb.log({'Test Error Rate': round((1 - accuracy) * 100, 2)})

    run.finish()

if __name__ == '__main__':
    main()
