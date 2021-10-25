
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image as tfimage
import uuid
import pathlib
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

HOME_FOLDER = os.getcwd()

MODEL_EMOTION_RECOGNIZE_PATH = 'MobileNetV2_25.h5'
EMOTION_CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprise']
MODEL_EMOTION_RECOGNIZE = tf.keras.models.load_model(MODEL_EMOTION_RECOGNIZE_PATH)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def predict_emotion(image_path): 
    #INPUT: PATH OF AN FACE
    #OUTPUT: PREDICTION OF EMOTION  
    
    #read file
    image = tf.io.read_file(image_path) 
    
    # preprocess
    image = tf.io.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image/ 255.0
    img_array  = tf.expand_dims(image, axis=0) #predict nhận theo batch (1,224,224,3)
    
    # predict
    prediction = MODEL_EMOTION_RECOGNIZE.predict(img_array)
    prediction = prediction.reshape(5)
    index = prediction.argmax()
    proba = prediction[index]
    emotion = EMOTION_CLASS_NAMES[index]
    return emotion, "{:.2f}%".format(proba*100)


def save_file_to_tmp(img):
    isExist = os.path.exists('tmp')
    if not isExist:
        os.makedirs('tmp')
    tmp_path = os.path.normpath('tmp')
    os.chdir(tmp_path)
    id = uuid.uuid1()
    fname_tmp = str(id) + '.jpg' 
    cv2.imwrite(fname_tmp,img)
    print("Image saved successful: ",fname_tmp)
    os.chdir(HOME_FOLDER) # Trả về thư mục gốc
    tmp_path = 'tmp\\'+ fname_tmp
    return str(tmp_path)

def get_image(image):
    with mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5) as face_mesh:
                
                        # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                canvas = np.zeros((image.shape[0],image.shape[1],3), dtype='uint8')
                canvas.fill(255)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                                image=canvas,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                                image=canvas,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                        h, w, c = canvas.shape
                        cx_min=  w
                        cy_min = h
                        cx_max = cy_max = 0
                        for lm in face_landmarks.landmark:
                            cx, cy = int(lm.x * w),int(lm.y * h)
                            if cx < cx_min:
                                cx_min = cx
                            if cy < cy_min:
                                cy_min = cy
                            if cx > cx_max:
                                cx_max = cx
                            if cy > cy_max:
                                cy_max = cy
                        image_to_predict = canvas[max(cy_min,0):cy_max, max(cx_min,0):cx_max]
                      
                        return image_to_predict


def get_music(emotion, genre_choice):
    all = pd.read_csv('expression.csv')
    playlist_emotion = all[(all['emotion']== emotion) & (all['genre'] == genre_choice) ]['url'].to_list()
    return playlist_emotion 
  
  