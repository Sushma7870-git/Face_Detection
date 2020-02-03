# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:14:48 2020

@author: Lenovo
"""

# Importing all required packages 
import cv2 

import matplotlib.pyplot as plt  
  
# Read in the cascade classifiers for face and eyes 
face_cascade = cv2.CascadeClassifier('data\haarcascade_eye.xml') 
eye_cascade = cv2.CascadeClassifier('data\haarcascade_frontalface_alt.xml') 
    
# create a function to detect face 
def adjusted_detect_face(img): 
      
    face_img = img.copy() 
      
    face_rect = face_cascade.detectMultiScale(face_img,  
                                              scaleFactor = 1.2,  
                                              minNeighbors = 5) 
      
    for (x, y, w, h) in face_rect: 
        cv2.rectangle(face_img, (x, y),  
                      (x + w, y + h), (255, 255, 255), 10)
          
    return face_img 
    
# create a function to detect eyes 
def detect_eyes(img): 
      
    eye_img = img.copy()     
    eye_rect = eye_cascade.detectMultiScale(eye_img,  
                                            scaleFactor = 1.2,  
                                            minNeighbors = 5)     
    for (x, y, w, h) in eye_rect: 
        cv2.rectangle(eye_img, (x, y),  
                      (x + w, y + h), (255, 255, 255), 10)         
    return eye_img 
  
# Reading in the image and creating copies 
img = cv2.imread(r'Test_Image\face1.jfif') 
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_copy1 = img.copy() 
img_copy2 = img.copy() 
img_copy3 = img.copy() 

# Detecting the face  
face = adjusted_detect_face(img_copy1) 
 
eyes = detect_eyes(img_copy2) 
plt.imshow(eyes) 
eyes_face = adjusted_detect_face(img_copy3) 
eyes_face = detect_eyes(eyes_face) 
plt.xticks([])
plt.yticks([])
plt.imshow(eyes_face) 