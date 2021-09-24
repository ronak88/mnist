import pandas as pd
import os
import numpy as np
from joblib import dump, load   # for importing joblib model
from tensorflow.keras import utils, models  # for importing cnn .h5 model
from cv2 import cv2
import easyocr


# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# cnn_model = models.load_model("mnist_cnn_model3.h5")
# simple_model = load('mnist_model1.joblib')

reader = easyocr.Reader(['en'])
videoCaptureObject = cv2.VideoCapture(0)
# videoCaptureObject = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while (True):
    #     videoCaptureObject.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    #     videoCaptureObject.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    ret, frame = videoCaptureObject.read()
    image = cv2.rectangle(frame, (20, 5), (620, 49), (22, 246, 200), -1)  # yellow solid box
    image = cv2.rectangle(frame, (20, 5), (620, 49), (0, 0, 0), 1)  # black outline box
    image = cv2.rectangle(frame, (48, 115), (202, 272), (22, 246, 200), 3)  # yellow box
    image = cv2.rectangle(frame, (48, 115), (202, 272), (0, 0, 0), 1)  # black outline box
    image = cv2.rectangle(frame, (35, 285), (215, 310), (22, 246, 200), -1)  # yellow box
    image = cv2.rectangle(frame, (35, 285), (215, 310), (0, 0, 0), 1)  # black outline box
    image = cv2.putText(image, 'Place your text inside the Box', (28, 105), 4, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
    #     image = cv2.putText(image, 'MNIST Handwritten Digit Recognition', (37,35), 4,0.88,(0,0,255),2, cv2.LINE_AA)
    #     image = cv2.putText(image, 'MNIST Handwritten Digit Recognition', (36,35), font,1,(0,255,0),2, cv2.LINE_AA)
    image = cv2.putText(image, 'Automatic number-plate Recognition', (35, 35), 4, 0.88, (0, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, 'Your Digit =', (45, 302), 4, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite("NewPicture2.jpg", frame)
    img = cv2.imread("NewPicture2.jpg", 0)
    img = img[125:265, 60:190]
    thresh, temp = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)
    cv2.imwrite("temp.jpg", temp)
    resized = cv2.resize(255 - temp, (28, 28), interpolation=cv2.INTER_AREA)

    # # simple logistic regression multiclass classification
    # simple_model = load('mnist_model1.joblib')
    # test_img=resized.reshape(784)
    # a=simple_model.predict([test_img])

    # # CNN Deep learning model
    # test_img = resized.reshape(1,28,28,1)
    # test_img = utils.normalize(test_img,axis=1)
    # a=cnn_model.predict([test_img])
    # a= np.argmax(a)

    # image = cv2.putText(image,str(a), (163,302),4,0.5,(0,0,0),1, cv2.LINE_AA)

    # easyocr method
    a = reader.readtext('temp.jpg')
    # a= a[0][1]
    try:
        image = cv2.putText(image, str(a[0][1]), (140, 302), 4, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('MNIST Handwritten Digit Recognition', frame)
    except:
        image = cv2.putText(image, "?", (163, 302), 4, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('MNIST Handwritten Digit Recognition', frame)
    if (cv2.waitKey(1) & 0xFF == ord('c')):
        cv2.imwrite("NewPicture2.jpg", frame)
        videoCaptureObject.release()
        cv2.destroyAllWindows()
        break
    elif (cv2.waitKey(1) & 0xFF == ord('q')):
        videoCaptureObject.release()
        cv2.destroyAllWindows()
        break
