from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from datetime import datetime, date, timedelta
from imutils.video import VideoStream
from mlx90614 import MLX90614
from pyzbar import pyzbar
from smbus2 import SMBus
import face_recognition
import pandas as pd
import numpy as np
import schedule
import imutils
import time
import glob
import cv2
import os
import re
import requests
from tkinter import*
from tkinter import messagebox

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
        faces = f_name.split('/')
        for name in faces:
            if name.endswith('.jpg'):
                name = name.replace('.jpg', '')
                newnames.append(name)
    
    names = newnames
    print('working')
    dt = date.today() - timedelta(15)
    dt = dt.strftime("%d/%m/%Y")

    df1 = pd.read_csv('Attendance.csv')
    namesrec = df1['Name'].tolist() 
    
    df1.drop(df1[df1['EntryDate'] <= dt].index, inplace = True) 
    df1.to_csv('Attendance.csv',index=False)
    
    df2 = pd.read_csv('NoMask.csv')
    namesrec = df2['Name'].tolist() 
    
    df2.drop(df2[df2['EntryDate'] <= dt].index, inplace = True) 
    df2.to_csv('NoMask.csv',index=False)
    
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


# In[4]:


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

def idscanner(frame):

    qrname = ""
    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'dataset/faces/')
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

            print("New_{} written!".format(img_name))
            now = datetime.now()
            curdate= now.strftime("%d/%m/%Y")
            timenow = now.strftime('%H:%M:%S')
            payload = {'a1':str(name), 'a2':str(curdate), 'a3':str(timenow), 'a4':'NewVisitor'}
            r = requests.get("http://www.securitynet.ml/nvisitor.php", params=payload)
            cv2.waitKey(5)
            return
        else:
            return


# In[6]:


def markvisitor(entry, name):

    vname=name
    visitor = []
    now = datetime.now()
    curdate= now.strftime("%d/%m/%Y")
    timenow = now.strftime('%H:%M:%S')
    bus = SMBus(1)
    sensor = MLX90614(bus, address=0x5A)
    temp= sensor.get_object_1()
    bus.close()
    temp = "{.2f}%".format(temp)
    
    if float(temp) > 38.30:
        entry = "Denied"
        payload = {'a1':str(vname), 'a2':str(curdate), 'a3':str(timenow), 'a4':str(temp), 'a5':'High Temperature'}
        r = requests.get("http://www.securitynet.ml/dvisitor.php", params=payload, timeout=5)
        r.close()
        return entry
    
    df = pd.read_csv('Attendance.csv')
    visitor = df.query('Name == @vname and ExitTime=="inside"')
    entrytime= visitor['EntryTime']
    visitor = visitor.tail(1)
    if visitor.empty:
        
        names_list = list(df['Name'])
        df.loc[len(df)] = [vname, curdate, timenow, temp, 'inside', 'none']
        print('Welcome')
 
        payload = {'a1':str(vname), 'a2':str(curdate), 'a3':str(timenow), 'a4':str(temp)}
        r = requests.get("http://www.securitynet.ml/alvisitor.php", params=payload, timeout=5)
        print(r.url)
        r.close()
        
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
def marknomask(name):
    vname=name
    visitor = []
    now = datetime.now()
    curdate= now.strftime("%d/%m/%Y")
    timenow = now.strftime('%H:%M:%S')
    df = pd.read_csv('NoMask.csv')
    visitor = df.query('Name == @vname')
    
    mdate = datetime.strptime(curdate, '%d/%m/%Y')
    mtime = datetime.strptime(timenow, '%H:%M:%S')
    
    if visitor.empty:
        df.loc[len(df)] = [vname, curdate, timenow,'No-Mask']
        payload = {'a1':str(vname), 'a2':str(curdate), 'a3':str(timenow), 'a4':'', 'a5':'No Mask'}
        r = requests.get("http://www.securitynet.ml/dvisitor.php", params=payload, timeout=5)
        print(r.url)
        r.close()
    else:
        visitor = visitor.tail(1)
        vdate = visitor['EntryDate'].tolist()
        vtime = visitor['EntryTime'].tolist()
        vdate = vdate[-1]
        vtime = vtime[-1]
        vdate = datetime.strptime(vdate, '%d/%m/%Y')
        vtime = datetime.strptime(vtime, '%H:%M:%S')
        
        if mdate == vdate:
            if mtime >= vtime + timedelta(minutes=30):
                df.loc[len(df)] = [vname, curdate, timenow,'No Mask']
                payload = {'a1':str(vname), 'a2':str(curdate), 'a3':str(timenow), 'a4':'', 'a5':'No Mask'}
                r = requests.get("http://www.securitynet.ml/dvisitor.php", params=payload, timeout=5)
                print(r.url)
                r.close()
        elif mdate != vdate:
            df.loc[len(df)] = [vname, curdate, timenow,'No Mask']
            payload = {'a1':str(vname), 'a2':str(curdate), 'a3':str(timenow), 'a4':'', 'a5':'No Mask'}
            r = requests.get("http://www.securitynet.ml/dvisitor.php", params=payload, timeout=5)
            print(r.url)
            r.close()
        else:
            pass
    
    df.to_csv('NoMask.csv',index=False)


# In[16]:
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
        
        ret, frame = video_capture.read(0)

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
                entry = ''
                markvisitor(entry, name)
                if entry == 'Denied':
                    label = 'Your Temperature is too high, Entry is Denied'                    
                else:
                    label = 'Welcome to the society'
            
            elif label != 'Mask' and name != 'Unknown':
                label = 'Mask not worn, Entry Denied Till mask is worn.'
                marknomask(name)
            
            elif name == "Unknown":
                label = 'Show full face for Registration'
                print('Show full face for Registration')
                idscanner(frame)

            # include the probability in the label
            label = "{}, {}".format(name, label)

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
    
class Login:

    def __init__(self,root):
        self.root=root
        self.root.title("Login system for ASST")
        self.root.geometry("1199x600+100+50")
        self.root.resizable(False,False)
        #self.bg=ImageTk.PhotoImage(file="dataset/faces/elon-m.jpg")
        #self.bg_image=Label(self.root,image=self.bg).place(x=0,y=0,relwidth=1,relheight=1)

       
        Frame_login=Frame(self.root,bg="white")
        Frame_login.place(x=150,y=150,height=340,width=500)
        
        title=Label(Frame_login,text="Login here",font=("Impact",35,"bold"),fg="#d77337",bg="white").place(x=90,y=30)
        desc=Label(Frame_login,text="REGISTERED LOGIN ONLY",font=("Helvetica",15,"bold"),fg="#d25d17",bg="white").place(x=90,y=100)
        lbl_user=Label(Frame_login,text="Username",font=("Helvetica",15,"bold"),fg="gray",bg="white").place(x=90,y=140)
        self.txt_user=Entry(Frame_login,font=("Helvetica",15),bg="lightgray")
        self.txt_user.place(x=90,y=170,width=350,height=35)
       
        lbl_pass=Label(Frame_login,text="Password",font=("Helvetica",15,"bold"),fg="gray",bg="white").place(x=90,y=210)
        self.txt_pass=Entry(Frame_login,font=("Helvetica"),bg="lightgray")
        self.txt_pass.place(x=90,y=240,width=350,height=35)
        Login_btn=Button(self.root,text="Login",command=self.login_function,cursor="hand2",fg="white",bg="#d77337",font=("Helvetica",20)).place(x=300,y=470,width=180,height=40)
        
    def login_function(self):
        us = self.txt_user.get()
        pas = self.txt_pass.get()
        payload = {'a1':str(us), 'a2':str(pas)}
        r = requests.post("http://www.securitynet.ml/verif.php", params=payload, timeout=5)
        if r.status_code == 200:
            asst()
        elif self.txt_pass.get()=="" or self.txt_user.get()=="":
            messagebox.showerror("Eroor","All fields are required",parent=self.root)            
        else:
             messagebox.showerror("Error","Wrong id/Password",parent=self.root)

        
root = Tk()
obj = Login(root)
root.mainloop()