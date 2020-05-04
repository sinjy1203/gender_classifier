import cv2 as cv
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
model = load_model("gender.h5")

cap = cv.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if ret == False:
        continue

    cv.imshow("frame", frame)
    if cv.waitKey(1) == 32:
        img = cv.resize(frame, (150, 150))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.reshape(img, (1, 150, 150, 3))
        img = conv_base.predict(img)
        img = np.reshape(img, (1, 4*4*512))
        pred = model.predict(img)
        if pred <= 0.5:
            print("man")
        else:
            print("woman")
            # cv.imwrite("my.jpg", frame)
    elif cv.waitKey(1)%0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()