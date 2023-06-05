from keras.models import load_model
from time import sleep
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Users\mozay\Desktop\project2\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\mozay\Desktop\project2\model.h5')

#emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

emotion_labels = ['Kizgin','Tiksinmis','Korkmus','Mutlu','Notr','Uzgun','Saskin']

cap = cv2.VideoCapture(0)
