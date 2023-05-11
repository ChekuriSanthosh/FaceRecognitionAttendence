import face_recognition
import cv2
import numpy as np
import csv
import os
# import glob
from datetime import datetime

video_capture=cv2.VideoCapture(0)

santhosh_image=face_recognition.load_image_file("photos/santhosh.png")
santhosh_encoding=face_recognition.face_encodings(santhosh_image)[0]

rahul_image=face_recognition.load_image_file("photos/rahul.png")
rahul_encoding=face_recognition.face_encodings(rahul_image)[0]

nikhil_image=face_recognition.load_image_file("photos/nikhil.png")
nikhil_encoding=face_recognition.face_encodings(nikhil_image)[0]

koushik_image=face_recognition.load_image_file("photos/koushik.png")
koushik_encoding=face_recognition.face_encodings(koushik_image)[0]


known_face_encoding=[
    santhosh_encoding,
    rahul_encoding,
    nikhil_encoding,
    koushik_encoding
]

known_faces_names=[
    "santhosh",
    "rahul",
    "nikhil",
    "koushik"
]

students=known_faces_names.copy()

face_locations=[]
face_encodings=[]
face_names=[]
s=True


now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(current_date+'.csv','w+',newline='')
lnwriter=csv.writer(f)

while True:
    _,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_faces_names[best_match_index]
            
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()