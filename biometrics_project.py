
# Codeing implementation of the Multimodal fusion of IRIS and FINGERPRINT

# Import necessary libraries
from google.colab import drive  # Import Google Drive library to mount the drive
drive.mount('/content/drive')  # Mount Google Drive to access files

import os  # Import os for file and directory operations
import cv2  # Import OpenCV for image processing
import numpy as np  # Import numpy for numerical operations
import tensorflow as tf  # Import TensorFlow for deep learning
from tensorflow.keras import layers, models, optimizers  # Import Keras components from TensorFlow
from tensorflow.keras.applications import VGG16  # Import VGG16 pre-trained model from Keras applications
from sklearn.model_selection import train_test_split  # Import train_test_split for data splitting

# Function to list the contents of a directory
def list_directory_contents(directory):
    if os.path.exists(directory):  # Check if the directory exists
        print(f"Contents of {directory}:")  # Print the directory path
        for item in os.listdir(directory):  # Iterate through the directory contents
            print(item)  # Print each item in the directory
    else:
        print(f"Directory does not exist: {directory}")  # Print error message if directory does not exist

# Function to load images from a person's directory
def load_images_from_person(person_dir):
    # Define paths for fingerprint, left iris, and right iris directories
    fingers_dir = os.path.join(person_dir, '1_Fingerprint')
    left_iris_dir = os.path.join(person_dir, '2_Iris_left')
    right_iris_dir = os.path.join(person_dir, '3_Iris_right')

    # Load fingerprint images
    fingerprint_images = []
    for i in range(1, 12):  # Assume 11 fingerprint images per person
        finger_image_path = os.path.join(fingers_dir, f'{i}.BMP')  # Define image path
        print(finger_image_path)  # Print image path
        finger_image = cv2.imread(finger_image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if finger_image is not None:
            fingerprint_images.append(finger_image)  # Append image to list if it exists
        else:
            print(f"Missing or invalid fingerprint image: {finger_image_path}")  # Print error if image is missing

    # Load left iris images
    left_iris_images = []
    for iris_id in range(1, 7):  # Assume 6 left iris images per person
        left_iris_image_path = os.path.join(left_iris_dir, f'{iris_id}.BMP')  # Define image path
        print(left_iris_image_path)  # Print image path
        left_iris_image = cv2.imread(left_iris_image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if left_iris_image is not None:
            left_iris_images.append(left_iris_image)  # Append image to list if it exists
        else:
            print(f"Missing or invalid left iris image: {left_iris_image_path}")  # Print error if image is missing

    # Load right iris images
    right_iris_images = []
    for iris_id in range(1, 7):  # Assume 6 right iris images per person
        right_iris_image_path = os.path.join(right_iris_dir, f'{iris_id}.BMP')  # Define image path
        print(right_iris_image_path)  # Print image path
        right_iris_image = cv2.imread(right_iris_image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
        if right_iris_image is not None:
            right_iris_images.append(right_iris_image)  # Append image to list if it exists
        else:
            print(f"Missing or invalid right iris image: {right_iris_image_path}")  # Print error if image is missing

    # Return loaded images
    return fingerprint_images, left_iris_images, right_iris_images

# Function to preprocess images
def preprocess_image(image, resize_dim=(64, 64)):
    if image is None:
        raise ValueError("Invalid image provided for preprocessing.")  # Raise error if image is invalid
    image = cv2.resize(image, resize_dim)  # Resize image to specified dimensions
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale image to BGR (3 channels)
    image = image / 255.0  # Normalize image to range [0, 1]
    return image  # Return preprocessed image

# Function to extract features using a model
def extract_features(image, model):
    image = np.expand_dims(image, axis=0)  # Add batch dimension to image
    features = model.predict(image)  # Extract features using the model
    return features.flatten()  # Flatten and return the extracted features

# Function to load the dataset from a directory
def load_dataset(dataset_dir):
    persons_data = []  # Initialize list to store data for each person
    for person_id in range(1, 45):  # Assume there are 44 persons in the dataset
        person_dir = os.path.join(dataset_dir, f'person{person_id}')  # Define person's directory path
        if os.path.isdir(person_dir):  # Check if directory exists
            fingerprint_images, left_iris_images, right_iris_images = load_images_from_person(person_dir)  # Load images
            persons_data.append({
                'fingers': fingerprint_images,  # Add fingerprint images to person's data
                'left_iris': left_iris_images,  # Add left iris images to person's data
                'right_iris': right_iris_images  # Add right iris images to person's data
            })
        else:
            print(f"Missing directory: {person_dir}")  # Print error if directory is missing
    return persons_data  # Return data for all persons

# Function to prepare data for training and testing
def prepare_data(persons_data, vgg_model):
    iris_data = []  # Initialize list to store iris features
    fingerprint_data = []  # Initialize list to store fingerprint features

    # Preprocess and extract features from images
    for person in persons_data:
        for left_iris, right_iris, fingers in zip(person['left_iris'], person['right_iris'], person['fingers']):
            preprocessed_left_iris = preprocess_image(left_iris)  # Preprocess left iris image
            preprocessed_right_iris = preprocess_image(right_iris)  # Preprocess right iris image
            preprocessed_fingers = [preprocess_image(finger) for finger in fingers]  # Preprocess fingerprint images

            iris_features = extract_features(preprocessed_left_iris, vgg_model)  # Extract features from left iris
            iris_features = np.concatenate((iris_features, extract_features(preprocessed_right_iris, vgg_model)))  # Concatenate left and right iris features

            fingerprint_features = np.array([extract_features(finger, vgg_model) for finger in preprocessed_fingers]).flatten()  # Extract and flatten fingerprint features

            iris_data.append(iris_features)  # Add iris features to list
            fingerprint_data.append(fingerprint_features)  # Add fingerprint features to list

    # Handle case where no data is loaded
    if len(iris_data) == 0 or len(fingerprint_data) == 0:
        print("Error: No data loaded. Check the dataset directory and structure.")  # Print error if no data is loaded
        return None, None, None, None  # Return None if data loading failed

    iris_data = np.array(iris_data)  # Convert iris data to numpy array
    fingerprint_data = np.array(fingerprint_data)  # Convert fingerprint data to numpy array

    # Combine features and create labels
    combined_features = np.hstack((iris_data, fingerprint_data))  # Horizontally stack iris and fingerprint features
    num_persons = len(iris_data)  # Get the number of persons
    labels = np.zeros((num_persons, 1))  # Initialize labels with zeros
    for i in range(num_persons):
        labels[i] = 1 if i % 2 == 0 else 0  # Create labels (1 for even indices, 0 for odd indices)

    # Split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(combined_features, labels, test_size=0.2, random_state=42)  # Split data
    return train_features, train_labels, test_features, test_labels  # Return training and testing data

# Function to define the Siamese network architecture
def siamese_network(input_shape):
    input_iris = layers.Input(shape=(input_shape,), name='input_iris')  # Define input for iris features
    input_fingerprint = layers.Input(shape=(input_shape,), name='input_fingerprint')  # Define input for fingerprint features

    # Function to create base network for feature extraction
    def create_base_network(input_shape):
        return models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),  # Dense layer with 128 units and ReLU activation
            layers.Dropout(0.3),  # Dropout layer with 0.3 dropout rate
            layers.Dense(64, activation='relu'),  # Dense layer with 64 units and ReLU activation
            layers.Dropout(0.3),  # Dropout layer with 0.3 dropout rate
            layers.Dense(32, activation='relu')  # Dense layer with 32 units and ReLU activation
        ])

    base_network = create_base_network(input_shape)  # Create base network

    iris_features = base_network(input_iris)  # Extract features from iris input
    fingerprint_features = base_network(input_fingerprint)  # Extract features from fingerprint input

    concatenated_features = layers.Concatenate(axis=-1)([iris_features, fingerprint_features])  # Concatenate extracted features

    dense1 = layers.Dense(128, activation='relu')(concatenated_features)  # Dense layer with 128 units and ReLU activation
    dropout1 = layers.Dropout(0.3)(dense1)  # Dropout layer with 0.3 dropout rate
    dense2 = layers.Dense(64, activation='relu')(dropout1)  # Dense layer with 64 units and ReLU activation
    dropout2 = layers.Dropout(0.3)(dense2) 
    dense3 = layers.Dense(32, activation='relu')(dropout2)  # Dense layer with 32 units and ReLU activation

    output = layers.Dense(1, activation='sigmoid')(dense3)  # Output layer with 1 unit and sigmoid activation

    siamese_model = models.Model(inputs=[input_iris, input_fingerprint], outputs=output)  # Create Siamese network model

    return siamese_model  # Return Siamese network model

# Function to train the Siamese network
def train_model(siamese_model, train_features, train_labels, batch_size=32, epochs=50):
    history = siamese_model.fit(
        [train_features[:, :train_features.shape[1] // 2], train_features[:, train_features.shape[1] // 2:]],  # Split features into iris and fingerprint inputs
        train_labels,  # Training labels
        batch_size=batch_size,  # Batch size for training
        epochs=epochs,  # Number of epochs for training
        validation_split=0.2  # Validation split
    )
    return history  # Return training history

# Function to evaluate the Siamese network
def evaluate_model(siamese_model, test_features, test_labels):
    loss, accuracy = siamese_model.evaluate(
        [test_features[:, :test_features.shape[1] // 2], test_features[:, test_features.shape[1] // 2:]],  # Split features into iris and fingerprint inputs
        test_labels  # Testing labels
    )
    print("Test Loss:", loss)  # Print test loss
    print("Test Accuracy:", accuracy)  # Print test accuracy

# Main function to execute the complete code
def main():
    drive.mount('/content/drive')  # Mount Google Drive
    dataset_dir = "/content/drive/MyDrive/RENAMED"  # Define dataset directory path

    # List the contents of the main directory
    list_directory_contents(dataset_dir)

    # List the contents of each person directory (for first 10 persons)
    for i in range(1, 11):
        list_directory_contents(os.path.join(dataset_dir, f'{i}'))

    # Load the dataset
    persons_data = load_dataset(dataset_dir)

    # Check if data is loaded successfully
    if not persons_data:
        print("No data found in dataset directory.")
        return

    # Load pre-trained VGG16 model for feature extraction
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    vgg_model = models.Model(inputs=vgg_model.input, outputs=vgg_model.layers[-1].output)  # Remove top layers to get feature extractor

    # Prepare data for training and testing
    train_features, train_labels, test_features, test_labels = prepare_data(persons_data, vgg_model)

    # Check if data preparation was successful
    if train_features is None or test_features is None:
        print("Data preparation failed.")
        return

    input_shape = train_features.shape[1] // 2  # Define input shape for the network (half of the total features)
    siamese_model = siamese_network(input_shape)  # Create Siamese network model

    # Compile the Siamese network model
    siamese_model.compile(
        loss='binary_crossentropy',  # Define loss function
        optimizer=optimizers.Adam(learning_rate=0.001),  # Define optimizer with learning rate
        metrics=['accuracy']  # Define metrics
    )

    # Train the Siamese network model
    history = train_model(siamese_model, train_features, train_labels)

    # Evaluate the Siamese network on the test data
    evaluate_model(siamese_model, test_features, test_labels)

if __name__ == "__main__":
    main()
