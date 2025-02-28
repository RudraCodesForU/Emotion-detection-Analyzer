import cv2
from deepface import DeepFace
# deepface is a FER script (FER : FACE EMOTION RECOGNITION)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'data.xml')
# data check
cap = cv2.VideoCapture(1)
# Checking WebCam
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open WebCam !")

while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(frame , actions = ['emotion'])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces = faceCascade.detectMultiScale(gray,1.1,4)

    #Capture_Box
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255 ,0), 2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    #User putText() method for inserting text in video
    cv2.putText(frame,
                result['dominant_emotion'],
                (50,50),
                font, 3,
                (0,0,255),
                2,
                cv2.LINE_4)
    cv2.imshow('Original video',frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()