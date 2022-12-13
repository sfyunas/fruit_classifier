#########################################################################
#                Convolutional Neural Network (CNN) 
#                             for 
#                   Fruit Image Classification
#             -----------------------------------------
#   
#  Comment: Finding optimal network architecture parameters using Keras Tuner
#
#   Author: Syed Fahad Yunas
#   Date:   Dec 11, 2022
#########################################################################


#------------------------------------------------------------------------
# Importing the required dependencies
#------------------------------------------------------------------------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters

import os

#------------------------------------------------------------------------
# Setting up dataflow for the training & validation images dataset
#------------------------------------------------------------------------

# 1. Data flow parameters

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'

batch_size = 32   # -- mini batch size; typical value is 32

img_width = 128
img_height = 128
num_channels = 3  # -- RGB channels

num_classes = 6  # -- In this task, we aim to classify a fruit image between 6 fruit classes

# 2. Image generators: applying transformations on training set

training_generator = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5, 1.5),
                                        fill_mode = 'nearest')

validation_generator = ImageDataGenerator(rescale = 1./255)


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
# Architecture tuning
#------------------------------------------------------------------------

def build_model(hp):
    model = Sequential()

    # 1a. Testing the number of filters for our first Convolutional layer
   
    model.add(Conv2D(filters = hp.Int("Input_Conv_filters", min_value = 32, max_value = 256, step = 32), 
                     kernel_size = (3,3), 
                     strides = (1,1), 
                     padding = 'same', 
                     input_shape = (img_width, img_height, num_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    # 1b. Testing for number of additional Convolutional layers (after the first Convolutional
    #     layer).
    # --> For each convolutional layer, the number of filters will also be checked for.
    
    for i in range(hp.Int("n_Conv_Layers", min_value = 1, max_value = 6, step = 1)):
        model.add(Conv2D(filters = hp.Int(f"Conv_{i}_Filters", min_value = 32, max_value = 288, step = 32),
                         kernel_size = (3,3), 
                         strides = (1,1), 
                         padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))

    # 1c. Flatten the output of the max pooling layer of block 2
    
    model.add(Flatten())

    # 1d. Testing for number of Dense layers and the number of neurons in each dense layer
    
    for j in range(hp.Int("n_Dense_Layers", min_value = 1, max_value = 4, step = 1)):  
        model.add(Dense(hp.Int(f"Dense{j}_Neurons", min_value = 32, max_value = 288, step = 32)))
        model.add(Activation('relu'))
        if hp.Boolean("Dropout"):
            model.add(Dropout(0.5))
    
        # 1e. Output layer
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))  # returns probability of each class

    # 2. Compiling the CNN network

    model.compile(loss = 'categorical_crossentropy',
                  optimizer = hp.Choice('Optimizer', values = ['adam', 'RMSProp']),
                  metrics = ['accuracy'])
   return model

tuner = RandomSearch(hypermodel = build_model,
                     objective = 'val_accuracy',
                     max_trials = 5,
                     executions_per_trial = 2,
                     directory = 'tuner_results',
                     project_name = 'fruit_tuned_cnn',
                     overwrite = True)


tuner.search(x = training_set,
             validation_data = validation_set,
             epochs = 50,
             batch_size = 32)

# Summary from the tuning search
tuner.result_summary()

# Best network (in terms of hyperparameters)
tuner.get_best_hyperparameters()[0].values

# Best network summary
tuner.get_best_models()[0].summary()

'''
After running the tuner for finding the best hyper-parameter, we have the following
results:
    The model has one input convolutional layer with 32 filtere
    The model has five intermediate convolutional layers with 96, 192, 64, 224, 192
    filters, respectively. 
    The model has one dense layer with 256 neurons and no drop out
    The model's output (dense) layer has 6 neurons.
    
    The model uses 'adam' as the optimizer 
    
    Based on these hyper parameters we will create a cnn network architecture

'''
#------------------------------------------------------------------------
# Network Architecture
#------------------------------------------------------------------------

# 1. Specifying the Convolutional Neural Network (CNN) architecture
#    based on the tuned hyper-parameters given above.

model = Sequential()

# 1a. Convolutional layer block 1
model.add(Conv2D(filters = 32, 
                 kernel_size = (3,3), 
                 strides = (1,1), 
                 padding = 'same', 
                 input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# 1b. Convolutional layer block 2
model.add(Conv2D(filters = 96,
                 kernel_size = (3,3), 
                 strides = (1,1), 
                 padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# 1c. Convolutional layer block 3
model.add(Conv2D(filters = 192,
                 kernel_size = (3,3), 
                 strides = (1,1), 
                 padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# 1d. Convolutional layer block 4
model.add(Conv2D(filters = 64,
                 kernel_size = (3,3), 
                 strides = (1,1), 
                 padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# 1e. Convolutional layer block 5
model.add(Conv2D(filters = 224,
                 kernel_size = (3,3), 
                 strides = (1,1), 
                 padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# 1f. Convolutional layer block 6
model.add(Conv2D(filters = 192,
                 kernel_size = (3,3), 
                 strides = (1,1), 
                 padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# 2. Flatten the output of the max pooling layer of block 6
model.add(Flatten())

# 3a. Dense layer 1
model.add(Dense(256))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# 3b. Output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))  # returns probability of each class

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

num_epochs = 50
model_filename = 'models/fruits_cnn_v03.h5'

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

model_filename = 'models/fruits_cnn_v03.h5'

img_width = 128
img_height = 128

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
    image = image * (1./255)
    
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
VGG16 model (pre-trained)             99%             99%              98%
Tuned CNN model (no data aug. &       97%             99%             100%                                
                 no drop out)
----------------------------------------------------------------------------
Comments (for later review):
    
From the figure and the performance stats, the tuned model performance comparable
to that of pre-trained VGG16 modeol which is quite amazing. We did not use image
data augmentation techniques or 

-- Signing off
----------------------------------------------------------------------------
'''