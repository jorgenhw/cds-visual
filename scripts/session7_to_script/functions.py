"""
## Importing the libraries
"""

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# TensorFlow and tf.keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Other
import numpy as np
import tqdm as tqdm # loading bar

# Function normalize and split data
def normalize_and_split_data(data, labels, test_size = 0.2):
    # normalise data
    data = data.astype("float")/255.0

    # split data
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                        labels, 
                                                        test_size=test_size)
    return X_train, X_test, y_train, y_test

def one_hot_encode(y_train, y_test):
    lb = LabelBinarizer() # The labelBinazier class is used to convert labels to one-hot encoding (flag variables) which is a format that is used to represent categorical data.
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return y_train, y_test, lb

def define_nn_model():
    # Define model
    model = Sequential() # Here I create my NN model using the Sequential class, meaning that it is a feed forward NN. With this model i've created i'm initializing a FW NN.
    model.add(Dense(256, # it's called dense, because the NN is fully connected. The first hidden layer has 256 nodes.
                     input_shape=(784,), # the input shape is 784 because the images are 28x28 pixels
                       activation="relu")) # The activation function is relu, which is a non-linear function that is used to introduce non-linearity into the model. 
    model.add(Dense(128, # Now we are adding second hidden layer, with 128 nodes
                     activation="relu")) #... with the relu function
    model.add(Dense(10, # Defining our output layer, with 10 nodes
                     activation="softmax")) # with a softmax function, which is used to normalize the output of a NN to a probability distribution over predicted output classes.
                                       # We are using the softmax, because we are dealing with a classification problem where we want the probability of each class.
    return model

def compile_model(model, loss = "categorical_crossentropy", metrics = ["accuracy"]):
    # Compile model
    sgd = SGD(0.01)
    model.compile(loss=loss, # The loss function is the categorical cross-entropy, which is used to measure the error between the predicted and the actual values.
                   optimizer=sgd, # stofastic gradient descent, which is an optimization algorithm used to minimize the loss function. 0.01 is the learning rate, 
                   metrics=metrics) # The metrics is the accuracy, which is used to evaluate the performance of the model.
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs = 10, batch_size = 32):
    # Train model
    history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=epochs, 
                    batch_size=batch_size) # For every 32 images we are updating the weights of the NN.
    return model, history

def visualize_training(history):
    # Visualize training progress
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    plt.figure()
    N = 10
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()

# Get classification report
def get_classification_report(model, X_test, y_test, lb):
    predictions = model.predict(X_test, batch_size=32)
    clas_report = classification_report(y_test.argmax(axis=1), 
                            predictions.argmax(axis=1), 
                            target_names=[str(x) for x in lb.classes_])
    print(clas_report)