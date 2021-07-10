import pytesseract
import cv2
import numpy as np
import time
from models import EasyOcr
from utils import warp_image, show_image


# img = cv2.imread("/home/ea/projects/SIRIUS21/data/1_7b8pORMeluIx7XG-smavDQ.png")
img = cv2.imread("photo_2021-07-10_12-12-37.jpg")
# img = cv2.imread("photo_2021-07-10_12-18-24.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

v = "./samples/numberOCR.mp4"

cap = cv2.VideoCapture(v)

model = EasyOcr()

while True:

    _, img = cap.read()

    warped_img = img
    # warped_img = warp_image(img, np.array(eval(str([(215, 686), (788, 468),(789, 682), (290, 953)])),dtype="float32"))

    results = model.predict(warped_img)
    for (bbox, text, prob) in results:

        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        cv2.rectangle(warped_img, tl, br, (0, 255, 0), 2)
        cv2.putText(warped_img, text, (tl[0], tl[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    show_image(warped_img)


# results = reader.readtext(image)

# cv2.imwrite("/home/ea/projects/SIRIUS21/ocr.pytorch/test_images/tt.png", warped_img)

# warped_img = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)

# warped_img = enlarge_img(warped_img, 200)

# print(pytesseract.image_to_string(warped_img, config = r'--psm 8'))
#
# show_image(warped_img)

