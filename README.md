# MusicRecommendationBasedOnFaceExpression
Please read pdf file to understand all the steps I take to complete this project:
https://github.com/GiangVu0912/MusicRecommendationBasedOnFaceExpression/blob/main/FINAL%20PROJECT.pdf

## Purposes and Motivations
People tend to express their emotions, mainly by their facial expressions. Music has always been known to alter the mood of an individual. Capturing and recognizing the emotion being voiced by a person and displaying appropriate songs matching the one's mood and can increasingly calm the mind of a user and overall end up giving a pleasing effect. The project aims to capture the emotion expressed by a person through facial expressions. A music player is designed to capture human emotion through the web camera interface available on computing systems. The software captures the image of the user and then with the help of image segmentation and image processing techniques extracts features from the face of a target human being and tries to detect the emotion that the person is trying to express. The project aims to lighten the mood of the user, by playing songs that match the requirements of the user by capturing the image of the user. Since ancient times the best form of expression analysis known to humankind is facial expression recognition. The best possible way in which people tend to analyze or conclude the emotion or the feeling or the thoughts that another  person is trying to express is by facial expression. In some cases, mood alteration may also help in overcoming situations like depression and sadness. With the aid of expression analysis, many health risks can be avoided, and also there can be steps taken that help brings the mood of a user to a better stage.

## Steps:
- Preprocess Data
- Train Model to predict
- Build app on Streamlit

### Preprocess Image Data:
- Dataset is a part of AffectNetDataset, include: 420.000 images and 8 class of expression
- Preprocess:\
    Split a single image folder into 8 expression folders from the annotation file\
    Filter out 5 classes each class 15,000 photos: angry, happy, neutral, surprise, sad\
    Extract data from photo to facemesh on white canvas (using MediaPipe Facemesh)
 
### Preprocess Music Data:
- Dataset contain 1 file csv name music_data.csv and 400 song of mp3
- File csv contain these information:\
Id of the music file\
Genre of the music file\
9 annotations by the participant (whether emotion was strongly felt for this song or not). 1 means emotion was felt.\
Participant's mood prior to playing the game, ranging from 1 to 5, 1 is really bad and 5 is really good\
Liking (1 if participant decided to report he liked the song).
- Preprocess:\
Filter out the songs with the mood before listening to the music corresponding to 5 emotions and have a high average like score and have a feeling about the song that matches the emotion. For example: the emotion is angry, choose mood = 1, likeness > 0.5, calmness > 0.5\
Upload mp3 files to Soundcloud\
Scrape data from Soundcloud to get the link of the song.

### Train Model

