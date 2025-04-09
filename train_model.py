import cv2
import csv
import glob
import os
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
import joblib
from sklearn.model_selection import train_test_split


header = ["label"]
for i in range(0, 784):
    header.append("pixel"+str(i))
with open('dataset.csv', "a") as f:
        writer = csv.writer(f)
        writer.writerow(header)

for label in range(10):
    dirList = glob.glob("captured_images/" + str(label) + "/*.png")

    for img_path in dirList:
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
        roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

        data = []
        data.append(label)
        rows, cols = roi.shape
        for i in range(rows):
            for j in range(cols):
                k = roi[i, j]
                if k > 100:
                    k = 1
                else:
                    k = 0
                data.append(k)
        with open('dataset.csv', "a") as f:
            writer = csv.writer(f)
            writer.writerow(data)

######################################################

data = pd.read_csv('dataset.csv')
data = shuffle(data)
X = data.drop(["label"], axis=1)
Y = data["label"]

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

classifier = SVC(kernel="linear", random_state=6)
classifier.fit(train_x, train_y)
joblib.dump(classifier, "model/digit_recognizer")

prediction = classifier.predict(test_x)
print("Accuracy: ", metrics.accuracy_score(prediction, test_y))


"""
This was for when i was capturing the images manually, now i'm using a dataset

def one_time_code():
    import pyscreenshot as ImageGrab
    import time

    images_folder = "captured_images//"
    for i in range(0, 50):
        time.sleep(8)
        image = ImageGrab.grab(bbox=(150, 280, 690, 850)) # these are the coordinates where it is taking a pic of my screen
        print("Saved..........",i)
        image.save(images_folder+str(i)+".png")
        print("clear screen now and redraw now")
"""
