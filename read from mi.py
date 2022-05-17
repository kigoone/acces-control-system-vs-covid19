import glob
import cv2 as cv
import numpy as np
haar_data = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
path = glob.glob("C:/Users/MN/Desktop/test/with_mask/*.jpg")
cv_img = []
for img in path:
    n = cv.imread(img)
    faces = haar_data.detectMultiScale(img)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (25, 10, 255), 4)
        face = img[y:y + h, x:x + w, :]
        face = cv.resize(face, (50, 50))
        print(len(cv_img))
        if len(cv_img) < 90:
            cv_img.append(face)

        cv.imshow('result', img)
        if cv.waitKey(2) == 27 or len(cv_img) >= 80:
            var = False
cv.destroyAllWindows()
np.save("with-mask.npy", cv_img)

