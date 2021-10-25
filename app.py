from enum import unique
import cv2
import streamlit as st
import mediapipe as mp
import random
import numpy as np
from streamlit_player import st_player, _SUPPORTED_EVENTS
from function import save_file_to_tmp, get_image, predict_emotion, get_music
from function import mp_drawing, mp_face_mesh, mp_drawing_styles, drawing_spec
import io
import inspect, os.path
from pathlib import Path

st.sidebar.image('https://media.giphy.com/media/tqfS3mgQU28ko/giphy.gif')
menu = ['Live Prediction', 'Take a picture']
choice = st.sidebar.selectbox('Live prediction or Take a picture', menu)

if choice=='Live Prediction':
    st.title("Live Prediction")
    run = st.checkbox('Show Camera')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while run:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            canvas = np.zeros((image.shape[0],image.shape[1],3), dtype='uint8')
            canvas.fill(255)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
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
                    img_path = save_file_to_tmp(image_to_predict)
                    emotion, proba = predict_emotion(img_path)
                    image = cv2.putText(image, emotion + '-' + proba,(cx_min,cy_min), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255,0,0), 1,  cv2.LINE_AA)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(image)
        else:
            st.write('Stopped')
            cap.release()
            

if choice == 'Take a picture':
    st.title("Take a picture")
    
    
    run = st.checkbox('Show Camera')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    picture = st.button('Click here to take picture')
    genre = ['classical', 'rock', 'electronic', 'pop']
    genre_choice = st.selectbox('Please choose your favorite genre music', genre)
    music = st.button('Click here to hear music')

    if picture:
        success, image = cap.read()
        image_to_predict = get_image(image)
        img_path = save_file_to_tmp(image_to_predict)
        emotion, proba = predict_emotion(img_path)
        image = cv2.putText(image, 'You might be ' + emotion, (20,20), cv2.FONT_HERSHEY_SIMPLEX,  0.8, (255,0,0), 1,  cv2.LINE_AA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image)

    all_path = sorted(Path('tmp').iterdir(), key=os.path.getmtime)
    if music:
        if all_path == []:
            emotion= 'angry'
        else:
            path_last = all_path[-1]
            emotion, proba = predict_emotion(str(path_last))
        st.write('You might be {}'.format(emotion))
        playlist_emotion = get_music(emotion, genre_choice)
        song = random.choice(playlist_emotion)
        st_player(song)

    while run:
        success, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(image)
    else:
        cap.release()