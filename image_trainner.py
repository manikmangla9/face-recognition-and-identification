import cv2
import os
import pickle
from PIL import Image
import numpy as np

base_dir=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(base_dir,"image")

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
current_id=1
label_id={}
y_label=[]
X_train=[]

for root,dirs,files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path=os.path.join(root,file)
			label=os.path.basename(root).replace(" ","_").lower()
			#print(label,path)
			if not label in label_id:
				label_id[label]=current_id
				current_id+=1
			id=label_id[label]
			pil_image=Image.open(path).convert("L")  #gray image
			image_arr=np.array(pil_image,"uint8")    #converting to numpy arr
			faces=face_cascade.detectMultiScale(image_arr,scaleFactor=1.05,minNeighbors=5)
			#detecting faces from image
			for (x,y,w,h) in faces:
				roi=image_arr[y:y+h,x:x+h]
				X_train.append(roi)
				y_label.append(id)


with open("labels.pickle",'wb') as f:
	pickle.dump(label_id,f)

recognizer.train(X_train,np.array(y_label))
recognizer.save("trainner.yml")

