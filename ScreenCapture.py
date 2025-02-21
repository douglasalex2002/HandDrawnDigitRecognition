import cv2
import csv
import glob
import os
import pandas as pd
from sklearn.utils import shuffle

# Define the header and the CSV file path
header = ["label"]
for i in range(784):
    header.append(f"pixel{i}")
csv_file_path = "dataset.csv"

# Check if the file exists and write the header if it does not
if not os.path.exists(csv_file_path):
    with open(csv_file_path, "w", newline='') as f:  # Ensure correct handling of newline
        writer = csv.writer(f)
        writer.writerow(header)

# Process images and write data to the CSV
for label in range(10):
    dirList = glob.glob(f"captured_images/{label}/*.png")

    for img_path in dirList:
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
        roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

        data = [label]
        rows, cols = roi.shape
        for i in range(rows):
            for j in range(cols):
                k = roi[i, j]
                data.append(1 if k > 100 else 0)

        with open(csv_file_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

# Load the data and display an example image
df = pd.read_csv(csv_file_path)
X = df.drop(["label"], axis=1)
Y = df["label"]

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

# Train a Support Vector Classifier
from sklearn.svm import SVC
from sklearn import metrics
import joblib

classifier = SVC(kernel="linear", random_state=6)
classifier.fit(train_x, train_y)
joblib.dump(classifier, "model/digit_recognizer")

# Evaluate the classifier
prediction = classifier.predict(test_x)
print("Accuracy: ", metrics.accuracy_score(prediction, test_y))

# Visualization of one image from the test set
import matplotlib.pyplot as plt

if not test_x.empty:
    img = test_x.iloc[0].values.reshape(28, 28)  # Reshape the first row of test_x
    label = test_y.iloc[0]
    plt.imshow(img, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()
else:
    print("Test set is empty.")

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
