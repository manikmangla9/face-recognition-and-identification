import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img =cv2.imread("m2.jpg")

#converting to grayimage
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#face coordinates
faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.1,minNeighbors=5)

#froming rectangular_face
for x,y,w,h in faces:
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)


#resized=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow("gray",img)
cv2.waitKey(20000)
cv2.destroyAllWindows()


