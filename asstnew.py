#!/usr/bin/env python
# coding: utf-8
 
# In[1]:
 
 
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from datetime import datetime, date, timedelta
from smbus2 import SMBus
from mlx90614 import MLX90614
import face_recognition
from pyzbar import pyzbar
import numpy as np
import pandas as pd
import schedule
import imutils
import time
import glob
import cv2
import os
import re
 
 
# In[8]:
 
 
def job():
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'dataset/faces/')
    list_of_files = [f for f in glob.glob(path+'*.jpg')]
    number_files = len(list_of_files)
 
    names = []
    newnames = []
    images = []
    f_names = list_of_files.copy()
    for f_name in f_names:
        faces = f_name.split('\\')
        for name in faces:
            if name.endswith('.jpg'):
                name = name.replace('.jpg', '')
                newnames.append(name)
    names = newnames
    print('working')
    dt = date.today() - timedelta(15)
    dt = dt.strftime("%d/%m/%Y")
 
    df = pd.read_csv('Attendance.csv')
    namesrec = df['Name'].tolist() 
    
    df.drop(df[df['EntryDate'] <= dt].index, inplace = True) 
    df.to_csv('Attendance.csv',index=False)
    
    for name in names:
        if name not in namesrec:
            file_path = os.path.join(cur_direc, 'dataset/faces/{}.jpg'.format(name))
            try:
                os.remove(file_path)
            except OSError as e:
                print("Error: %s : %s" % (file_path, e.strerror))
 
# schedule.every().day.at("20:17").do(job) 
# while True:
#     schedule.run_pending()
#     time.sleep(1)
 
 
# In[3]:
 
 
def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))
 
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    #print(detections.shape)
 
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
 
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
 
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
 
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
 
            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
 
            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))
 
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
 
    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)
 
 
# In[4]:
 
 
def idscanner(frame):
    #1
    # camera = cv2.VideoCapture(0)
    # ret, frame1 = camera.read()
    qrname = ""
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'dataset/faces/')
    #2
    # while ret:
    #     ret, frame1 = video_capture.read()
    barcodes = pyzbar.decode(frame)
    for barcode in barcodes:
        x, y , w, h = barcode.rect
        #1
        barcode_info = barcode.data.decode('utf-8')
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
        qrname = barcode_info
        #2
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, barcode_info, (x + 6, y - 6), font, 2.0, (255, 255, 255), 1)
        #3
        if qrname:
            print(qrname)
            qrname = qrname.splitlines()
            name = re.findall(r":(.*)",str(qrname[0]))
            name = ''.join(str(e) for e in name)
            #name = unknown_names[0]
            if name:
                print(name)
                img_name = "{name}.jpg".format(name=name)
                cv2.imwrite(str(path) + img_name, frame)
#                 cv2.destroyAllWindows()
            print("{} written!".format(img_name))
            cv2.waitKey(5)
            return
 
 
# In[5]:
 
 
def markvisitor(name):
    
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    temp= sensor.get_object_1()
    bus.close()
    
    vname=name
    df = pd.read_csv('Attendance.csv')
    now = datetime.now()
    visitor = []
    curdate= now.strftime("%d/%m/%Y")
    timenow = now.strftime('%H:%M:%S')
    
    visitor = df.query('Name == @vname and ExitTime=="inside"')
    entrytime= visitor['EntryTime']
    
    if visitor.empty:    
        names_list = list(df['Name'])
        df.loc[len(df)] = [vname, curdate, timenow, temp, 'inside', 'none']
        print('Welcome')
        if name in names_list:
            visitor = df.query('Name ==@name')
            exittime=visitor['ExitTime']        
    else:
        entrytime = pd.to_datetime(entrytime, format='%H:%M:%S')
        diff = str(datetime.strptime(timenow, '%H:%M:%S') - entrytime)
        dts = str(diff)
        match2 = re.findall(r":(.*):",dts)     
        
        try:
            if int(match2[0]) >= 2:
                vname1=str(name)
                #print(vname1)
                df.loc[df['Name'].str.contains(vname1), 'ExitTime'] = timenow                
                df.loc[df['Name'].str.contains(vname1), 'ExitTemp'] = temp
                print('Nice having you.')
                time.sleep(5)
        except:
            print("catch")
            pass    
 
    df.to_csv('Attendance.csv',index=False)
 
 
# In[7]:
 
 
def asst():
    
    faces_encodings = []
    faces_names = []
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'dataset/faces/')
    list_of_files = [f for f in glob.glob(path+'*.jpg')]
    number_files = len(list_of_files)
 
    names = []
    newnames = []
    images = []
    f_names = list_of_files.copy()
    for f_name in f_names:
        faces = f_name.split('\\')
        for name in faces:
            if name.endswith('.jpg'):
                name = name.replace('.jpg', '')
                newnames.append(name)
    names = newnames
 
    for i in range(number_files):
        globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
        globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
        faces_encodings.append(globals()['image_encoding_{}'.format(i)])
 
        # Create array of known names
        names[i] = names[i].replace(cur_direc,'')  
        faces_names.append(names[i])
 
 
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    prototxtPath = r"face_detector/deploy.prototxt"
    weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")
    
    schedule.every(15).days.at("00:30").do(job) 
    
    video_capture = cv2.VideoCapture(0)
    qrname =""
    while True:
        schedule.run_pending()
        
        ret, frame = video_capture.read()
 
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
 
        rgb_small_frame = small_frame[:, :, ::-1]
 
        if process_this_frame:
            face_locations = face_recognition.face_locations( rgb_small_frame)
            face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
 
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces (faces_encodings, face_encoding)
                name = "Unknown"
 
                face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = faces_names[best_match_index]
                    
                face_names.append(name)
        process_this_frame = not process_this_frame
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
 
        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            if label == 'Mask' and name != 'Unknown':
                label = 'Mask Detectected, Entry Permitted.'
                markvisitor(name)
            
            
            elif label != 'Mask' and name != 'Unknown':
                label = 'No Mask Detectected, Entry Denied.'
                print('Please put on a mask')
            
            
            elif name == "Unknown":
                print('Show full face for Registration')
                idscanner(frame)
                asst()
            # include the probability in the label
            label = "{}, {}: {:.2f}%".format(name, label, max(mask, withoutMask) * 100)
 
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    video_capture.release() 
    cv2.destroyAllWindows()
    
asst()