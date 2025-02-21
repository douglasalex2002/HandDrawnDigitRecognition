import joblib
import cv2
import pyscreenshot as ImageGrab
import pandas as pd


model=joblib.load("model/digit_recognizer")
images_folder="img/"

while True:
    img = ImageGrab.grab(bbox=(150, 280, 690, 850)) # these are the coordinates where it is taking a pic of my screen
    img.save(images_folder+"img.png")
    im = cv2.imread(images_folder+"img.png")
    im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (15,15), 0)

    ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

    rows, cols = roi.shape

    column_names = [f'pixel{i}' for i in range(0, 784)]
    X = []

    for i in range(rows):
            for j in range(cols):
                k = roi[i, j]
                X.append(1 if k > 100 else 0)

    X_df = pd.DataFrame([X], columns=column_names)

    predictions = model.predict(X_df)
    print("Prediction:", predictions[0])
    cv2.putText(im, "Prediction is: "+str(predictions[0]), (20, 20), 0, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.startWindowThread()
    cv2.namedWindow("Result")
    cv2.imshow("Result", im)
    cv2.waitKey(10000)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
