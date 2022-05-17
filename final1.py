import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

with_mask = np.load('with-mask.npy')
withoutm = np.load('without.npy')
with_mask = with_mask.reshape(200, 50 * 50 * 3)
withoutm = withoutm.reshape(200, 50 * 50 * 3)
x = np.r_[with_mask, withoutm]
labels = np.zeros(x.shape[0])
labels[199:] = 1.0
names={0:'Mask',1:'no Mask'}
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.20)
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
svm = SVC()
svm.fit(x_train, y_train)
x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)
haar_data = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv.VideoCapture(0)
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
            face= face.reshape(1,-1)
            face=pca.transform(face)
            result = svm.predict(face)
            n = names[int(result)]
            print(n)



            cv.imshow('result', img)
            if cv.waitKey(2) == 27:
                var = False
capture.release()
cv.destroyAllWindows()