import numpy as np
import glob
import cv2 as cv
haar_data = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
path = glob.glob("C:/Users/MN/Desktop/dataset/with_mask/*.png")
data=[]
for file in path:
   flag, img = cv.imread(file)

    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (25, 10, 255), 4)
            face = img[y:y + h, x:x + w, :]
            face = cv.resize(face, (50, 50))
            print(len(data))

                data.append(face)

np.save("with-mask-imported.npy", data)
