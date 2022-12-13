'''
    Utility functions for: 
        
        1. Preprocessing the input data (image)
        2. Passing the processed image to model for prediction
        
        
    Author: Syed Fahad Yunas
    Date: December 12, 2022
'''


# Importing dependency
from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize
import numpy as np
import os


# fruit labels list
labels_list = ['apple','avocado','banana', 'kiwi', 'lemon', 'orange']

# Preprocessing function: preprocesses the input image suitable for passing to 
# the image classifier model for prediction.
#   input  --> takes file path of the image
#   output --> proprocessed image

# The model was trained on images with following dimensions

img_width = 128
img_height = 128

def preprocess_image(filepath):
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)  # axis = 0 means that the additional dimension will be
                                             # added at the very start of the array
    image = image * (1./255)
    
    return image

# Image prediction function: returns the prediction label and predicted probability
#   input --> takes preprocessed image, and the classifier model
#   ouput --> provides predicted label along with predicted probability.

def make_prediction(image, model):
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    
    return predicted_label, predicted_prob

# Function to save uploaded file to disk
def save_uploadedfile(uploadedfile, dir_name = 'tmpCache'):
     with open(os.path.join(dir_name,uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return uploadedfile.name

