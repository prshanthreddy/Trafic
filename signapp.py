import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import tensorflow as tf
import pyttsx3

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',
            2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)',
            4:'Speed limit (70km/h)',
            5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)',
            7:'Speed limit (100km/h)',
            8:'Speed limit (120km/h)',
            9:'No passing',
            10:'No passing veh over 3.5 tons',
            11:'Right-of-way at intersection',
            12:'Priority road',
            13:'Yield',
            14:'Stop',
            15:'No vehicles',
            16:'Vehicle > 3.5 tons prohibited',
            17:'No entry',
            18:'General caution',
            19:'Dangerous curve left',
            20:'Dangerous curve right',
            21:'Double curve',
            22:'Bumpy road',
            23:'Slippery road',
            24:'Road narrows on the right',
            25:'Road work',
            26:'Traffic signals',
            27:'Pedestrians',
            28:'Children crossing',
            29:'Bicycles crossing',
            30:'Beware of ice/snow',
            31:'Wild animals crossing',
            32:'End speed + passing limits',
            33:'Turn right ahead',
            34:'Turn left ahead',
            35:'Ahead only',
            36:'Go straight or right',
            37:'Go straight or left',
            38:'Keep right',
            39:'Keep left',
            40:'Roundabout mandatory',
            41:'End of no passing',
            42:'End no passing vehicle > 3.5 tons' }

def image_processing(img):
    model = load_model('TSR.h5')
    image = Image.open(img)
    image = image.resize((30,30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    predict_x=model.predict(image)
    classes_x=np.argmax(predict_x,axis=1)
    sign = classes[int(classes_x)]
    return sign

st.title('Traffic Sign Recognition')
st.write('This is a simple image classification web app to predict traffic signs.')
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    # st.warning("Classifying...")
    label = image_processing(file)
    st.success('This image most likely belongs to {} with a {:.2f} percent confidence.'.format(label, 100))
    st.write('Done!')
    language = 'en'
    engine = pyttsx3.init()
    engine.setProperty('rate', 185)
    voice = engine.getProperty('voice')
    engine.say(label)
    engine.runAndWait()
