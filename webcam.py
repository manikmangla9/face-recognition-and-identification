import cv2
import pickle

label={}
with open("labels.pickle",'rb') as f:
   oglabel=pickle.load(f)
   label={v:k for k,v in oglabel.items()}

faceCascade = cv2.CascadeClassifier( "haarcascade_frontalface_default.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        roi_gray=gray[y:y+h,x:x+h]
        roi_color=frame[y:y+h,x:x+h]
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #prediction
        id,conf=recognizer.predict(roi_gray)
        if conf>=45 and conf<=85 :
            print(label[id])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=label[id]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()