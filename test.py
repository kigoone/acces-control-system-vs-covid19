import numpy as np
import cv2 as cv

haar_data = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture=cv.VideoCapture(0)
data = []
var = True
while var:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (25, 10, 255), 4)
            face = img[y:y + h, x:x + w, :]
            face = cv.resize(face, (50, 50))
            print(len(data))
            if len(data) < 400:
                data.append(face)

            cv.imshow('result', img)
            if cv.waitKey(2) == 27 or len(data) >= 200:
                var = False
capture.release()
cv.destroyAllWindows()
np.save("with-mask.npy", data)
