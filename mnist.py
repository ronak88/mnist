# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import linear_model,preprocessing
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.keras import utils,models
from joblib import load
from cv2 import cv2

cnn_model = models.load_model("mnist_cnn_model3.h5")
model = load('mnist_model1.joblib')
videoCaptureObject = cv2.VideoCapture(0)
while(True):
    #     videoCaptureObject.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    #     videoCaptureObject.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    ret, frame = videoCaptureObject.read()
    cv2.rectangle(frame, (20, 5), (620, 49), (22, 246, 200), -1)  # yellow solid box
    cv2.rectangle(frame, (20, 5), (620, 49), (0, 0, 0), 1)  # black outline box
    cv2.rectangle(frame, (48, 115), (202, 272), (22, 246, 200), 3)  # yellow box
    cv2.rectangle(frame, (48, 115), (202, 272), (0, 0, 0), 1)  # black outline box
    cv2.rectangle(frame, (48, 285), (202, 310), (22, 246, 200), -1)  # yellow box
    image = cv2.rectangle(frame, (48, 285), (202, 310), (0, 0, 0), 1)  # black outline box

    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, 'Place your text inside the Box', (28, 105), 4, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
    #     image = cv2.putText(image, 'MNIST Handwritten Digit Recognition', (37,35), 4,0.88,(0,0,255),2, cv2.LINE_AA)
    #     image = cv2.putText(image, 'MNIST Handwritten Digit Recognition', (36,35), font,1,(0,255,0),2, cv2.LINE_AA)
    image = cv2.putText(image, 'MNIST Handwritten Digit Recognition', (35, 35), 4, 0.88, (0, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, 'Your Digit =', (63, 302), 4, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite("NewPicture2.jpg", frame)
    img = cv2.imread("NewPicture2.jpg", 0)
    img = img[125:265, 60:190]
    thresh, temp = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)
    resized = cv2.resize(255 - temp, (28, 28), interpolation=cv2.INTER_AREA)
    test_img = resized.reshape(784)

    # # Sklearn Joblib Logistic Model 
    # a = model.predict([test_img])

    # Tensorflow CNN Model
    test_img = resized.reshape(1,28,28,1)
    test_img = utils.normalize(test_img,axis=1)
    a=cnn_model.predict([test_img])
    a= np.argmax(a)

    image = cv2.putText(image, str(a), (163, 302), 4, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('MNIST Handwritten Digit Recognition', frame)
    if(cv2.waitKey(1) & 0xFF == ord('c')):
        cv2.imwrite("NewPicture2.jpg", frame)
        videoCaptureObject.release()
        cv2.destroyAllWindows()
        break
    elif(cv2.waitKey(1) & 0xFF == ord('q')):
        videoCaptureObject.release()
        cv2.destroyAllWindows()
        break