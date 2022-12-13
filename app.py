'''
    Front-end for Fruit Image Classifier using streamlit library
    
    Author: Syed Fahad Yunas
    
    Date: 11 December 2022
'''

#----------------------------------------------
# Import dependencies
#----------------------------------------------

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import streamlit as st
import utility_functions as utl
from tempfile import NamedTemporaryFile


#----------------------------
# Loading the model pipeline
#----------------------------

model = load_model('models/fruits_cnn_v05.h5')

st.set_option('deprecation.showfileUploaderEncoding', False)



#------------------------------------
# Creating a web-page for our web-app
#------------------------------------

# Adding the title and subtitle and description
st.title("Fruit Image Classifier")
st.subheader("Please upload an image of fruit for image classification") 
st.write("Fruit classes currently supported by the model: Apple, Avocado, Banana, Kiwi, Orange)")

buffer = st.file_uploader(label = "Please upload an whole image of a fruit", 
                                type = ['jpg', 'jpeg'], 
                                accept_multiple_files=False)

temp_file = NamedTemporaryFile(delete=False)

if buffer is not None:
    temp_file.write(buffer.getvalue())
    st.image(temp_file.name)
    #pre-processing the image
    prc_img = utl.preprocess_image(temp_file.name)
    #passing the processed image to the model for classification
    pred_label, pred_proba = utl.make_prediction(prc_img, model)
    st.write(f"The model identifies the image as {pred_label} ({pred_proba:.0%})")
