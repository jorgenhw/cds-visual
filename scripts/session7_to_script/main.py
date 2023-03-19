from sklearn.datasets import fetch_openml
import functions as func
import argparse


def main(args):
    # Import data
    print("Loading data...")
    data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)

    # normalise data
    data = data.astype("float")/255.0

    # Function normalize and split data
    print("Normalizing and splitting data...")
    X_train, X_test, y_train, y_test = func.normalize_and_split_data(data, labels, test_size = args.test_size) #arg

    # One hot encoding the labels
    print("One hot encoding the labels...")
    y_train, y_test, lb = func.one_hot_encode(y_train, y_test)

    # Define model
    print("Defining model...")
    model = func.define_nn_model()

    # Compile the model
    print("Compiling model...")
    model = func.compile_model(model, 
                               loss = args.loss, # arg
                               metrics = args.metrics) # arg

    # Train the model
    print("Training model...")
    model, history = func.train_model(model, X_train, y_train, X_test, y_test, epochs = args.epocs, batch_size = args.batch_size) #(2xargs)

    # Visualize the training process
    print("Visualizing training process...")
    func.visualize_training(history)

    # Get classification report
    print("Getting classification report...")
    func.get_classification_report(model, X_test, y_test, lb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for training and evaluating a machine learning model visual classification using tensorflow.')
    parser.add_argument('--test_size', type=float, default=0.2, help='specify the size of the test/train split, default is 0.2')
    parser.add_argument('--loss', type=str, default="categorical_crossentropy", help='The loss function is the categorical cross-entropy, which is used to measure the error between the predicted and the actual values. Default is categorical_crossentropy')
    parser.add_argument('--metrics', type=list, default=["accuracy"], help='The metrics is the accuracy, which is used to evaluate the performance of the model. Default is accuracy')
    parser.add_argument('--epocs', type=int, default=10, help='The number of epochs is the number of times the model is trained on the entire dataset. Default is 10')
    parser.add_argument('--batch_size', type=int, default=32, help='For every 32 images we are updating the weights of the NN. Default is 32')
    args = parser.parse_args()
    main(args)