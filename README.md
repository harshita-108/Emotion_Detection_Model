
# Emotion Detector GUI
This project is a graphical user interface (GUI) application for detecting emotions from facial expressions in images. It utilizes a pre-trained deep learning model to classify emotions and provides a user-friendly interface for uploading images and displaying the detected emotions.
## Features
- Emotion Detection: Detects seven types of emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- User-Friendly Interface: Simple and intuitive GUI for uploading images and viewing results.
- Real-Time Processing: Quickly processes images and displays the detected emotion.

## Installation
### Prerequisites
- Python 3.6 or higher
- Required libraries: tkinter, numpy, opencv-python, tensorflow, Pillow, sklearn
### Clone the Repository
- git clone https://github.com/harshita-108/Emotion_Detection_Model.git
- cd Emotion_Detection_Model
### Install Dependencies
pip install -r requirements.txt
### Download Pre-trained Model and Haar Cascade
- Ensure you have the pre-trained model files model_a1.json and model_1.weights.h5 in the project directory.
- Download the Haar cascade for face detection (haarcascade_frontalface_default.xml) and place it in the project directory
## Usage
#### 1. Run the Application
- python gui.py on anaconda prompt in the environment which have all modules installed.
#### 2. Upload Image
- Click on the "Upload Image" button to select an image file from your system.
#### 3.Detect Emotion
- Once the image is uploaded, click on the "Detect Emotion" button to analyze the image and display the detected emotion.

## Model Training
The emotion detection model used in this project is trained on the FER-2013 dataset, which contains labeled facial expressions. The model architecture consists of convolutional and dense layers to effectively extract features and classify emotions.
### Training Code
The training code is provided below for reference. You can customize and train your own model using this code:
    def model_fer(input_shape):
    inputs = Input(input_shape)
    conv_1 = Convolution(inputs, 64, (3, 3))
    conv_2 = Convolution(conv_1, 128, (5, 5))
    conv_3 = Convolution(conv_2, 512, (3, 3))
    conv_4 = Convolution(conv_2, 512, (3, 3))
    
    flatten = Flatten()(conv_4)
    
    dense_1 = Dense_f(flatten, 256)
    dense_2 = Dense_f(dense_1, 512)
    
    output = Dense(7, activation="softmax")(dense_2)
    model = Model(inputs=[inputs], outputs=[output])
    
    opt = Adam(learning_rate=0.0005)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
    return model

## Acknowledgements

- FER-2013 Dataset: A publicly available dataset for facial expression recognition.
- Haar Cascades: Pre-trained classifiers for face detection provided by OpenCV.
- TensorFlow: An open-source platform for machine learning.
- Tkinter: The standard Python interface to the Tk GUI toolkit.
