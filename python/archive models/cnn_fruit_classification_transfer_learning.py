#########################################################################
#                Convolutional Neural Network (CNN) 
#                             for 
#                   Fruit Image Classification
#             -----------------------------------------
#   
#  Comment: Using pre-trained VGG16 for the fruit image classification task 
#           using transfer-learning 
#
#   Author: Syed Fahad Yunas
#   Date:   Nov 6, 2022
#########################################################################


#------------------------------------------------------------------------
# Importing the required dependencies
#------------------------------------------------------------------------

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


#------------------------------------------------------------------------
# Setting up dataflow for the training & validation images dataset
#------------------------------------------------------------------------

# 1. Data flow parameters

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'

batch_size = 32   # -- mini batch size; typical value is 32

img_width = 224   # -- The VGG16 model was originally trained on 224 x 224 pixels images
img_height = 224
num_channels = 3  # -- RGB channels

num_classes = 6   # -- In this task, we aim to classify a fruit image between 6 fruit classes

# 2. Image generators: applying transformations on training set

# -- Using the VGG16 preprocessing method to preprocess our image.
#    -> VGG16 uses a different preprocessing technique; The color 
#       channels are converted from RGB to BGR and zero-centered 
#       with respect to the ImageNet dataset


training_generator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5, 1.5),
                                        fill_mode = 'nearest')

validation_generator = ImageDataGenerator(preprocessing_function = preprocess_input)


# 3. Image flows

training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')


#------------------------------------------------------------------------
# Network Architecture
#------------------------------------------------------------------------

# 1. Loading the VGG16 pre-trained model
vgg = VGG16(input_shape = (img_width, img_height, num_channels),
            include_top = False)


# 2. Freezing the trainable parameters in the VGG16 architecture
#    (We want to use the pre-trained model - therefore we do not want to
#    to modify/update the VGG16 model parameters during the training process)

for layer in vgg.layers:
    layer.trainable = False

# 3. The Dense layers will receive input from the last layer of the VGG16 i.e.
#    the MaxPooling layer. Therefore, we want to flatten the input so that it
#    can be passed as input to our dense layer.

flatten = Flatten()(vgg.output)

# 4. Adding Dense layers (which will be trainable)
dense1 = Dense(128, activation = 'relu')(flatten)
dense2 = Dense(128, activation = 'relu')(dense1)

output = Dense(num_classes, activation = 'softmax')(dense2)

# 5. Using the functional API, we need to specify the input and output for our
#    model

model = Model(inputs = vgg.inputs, outputs = output)

# 2. Compiling the CNN network

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


# 3. Viewing the CNN architecture

model.summary()



#------------------------------------------------------------------------
# Training the CNN network
#------------------------------------------------------------------------

# 1. Setting up the training parameters

num_epochs = 10   # num_epochs = 50
model_filename = 'models/fruits_cnn_v04.h5'

# 2. Setting up callbacks fpr saving the best model

save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1,
                                  save_best_only = True)

# 3. Training the network

history = model.fit(x = training_set, 
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])


#------------------------------------------------------------------------
# Visualizing the training & validation performance
#------------------------------------------------------------------------

import matplotlib.pyplot as plt

# 1. Plotting the validation results

fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# 2. Getting the best epoch performance for validation accuracy

max(history.history['val_accuracy'])


#########################################################################
#           Make Predictions On New Data (Test Set)
#########################################################################

#------------------------------------------------------------------------
# Importing the required dependencies
#------------------------------------------------------------------------

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import numpy as np
import pandas as pd
from os import listdir


#------------------------------------------------------------------------
# Setting up the parameters for prediction and loading the saved model
#------------------------------------------------------------------------

model_filename = 'models/fruits_cnn_v04.h5'

img_width = 224
img_height = 224

labels_list = ['apple','avocado','banana', 'kiwi', 'lemon', 'orange']

model = load_model(model_filename)

#------------------------------------------------------------------------
# Defining functions for:
#   a. Loading the image from directory & applying pre-processing
#   b. Passing the image to the model for prediction
#------------------------------------------------------------------------

#-- Image pre-processing function

def preprocess_image(filepath):
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)  # axis = 0 means that the additional dimension will be
                                             # added at the very start of the array
    image = preprocess_input(image)
    
    return image

#-- Image prediction function

def make_prediction(image):
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    
    return predicted_label, predicted_prob


#------------------------------------------------------------------------
# Looping through the test dataset and applying the functions
#------------------------------------------------------------------------

test_source_dir = 'data/test'
folder_names = ['apple','avocado','banana', 'kiwi', 'lemon', 'orange']

actual_labels = []
predicted_labels_list = []
predicted_probabilities_list = []
filenames = []

for folder in folder_names:
    images = listdir(test_source_dir + '/' + folder)
    
    for image in images:
        processed_image = preprocess_image(test_source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_probability = make_prediction(processed_image)
        
        actual_labels.append(folder)
        predicted_labels_list.append(predicted_label)
        predicted_probabilities_list.append(predicted_probability)
        filenames.append(image)
  
        
#------------------------------------------------------------------------
# Creating a dataframe for further analysis
#------------------------------------------------------------------------

predictions_df = pd.DataFrame({"actual_label" : actual_labels,
                               "predicted_label" : predicted_labels_list,
                               "predicted_probability" : predicted_probabilities_list,
                               "file_name" : filenames})


#-- adding a binary column to the data frame to show whether predictions was correct or not    

predictions_df['prediction_correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_label'], 1, 0)


#------------------------------------------------------------------------
# Calculating the overall test set accuracy
#------------------------------------------------------------------------

test_set_accuracy = sum(predictions_df['prediction_correct'])/len(predictions_df)
print(test_set_accuracy)  # The test set accuracy on our basic network is ~ 72 %


#------------------------------------------------------------------------
# Creating a confusion matrix
#------------------------------------------------------------------------

# 1. Confusion matrix (Raw numbers)
confusion_matrix = pd.crosstab(predictions_df['predicted_label'], 
                                   predictions_df['actual_label'])
print(confusion_matrix)

#. Confusion matrix (Percentages)
confusion_matrix_prc = pd.crosstab(predictions_df['predicted_label'], 
                                   predictions_df['actual_label'],
                                   normalize = 'columns')

print(confusion_matrix_prc)


'''
(Model performance)
   
                                  Training set    Validation set    Test set
                                  ------------    --------------    --------    

Basic CNN Model (no dropout)         100%             82%              85%  
CNN Model (with 50% dropout)          95%             87%              92%
CNN Model (data aug./No dropout)      98%             94%              93%
VGG16 model                           99%            100%              97%
----------------------------------------------------------------------------
Comments (for later review): 
    
From the figure and the performance stats, the pre-trained VGG model with data
augmentation but trained for 50 epochs did show significant performance
improvement over our CNN model (with data augmentation only).
 
-- Signing off
----------------------------------------------------------------------------
'''