import cv2
import numpy as np
from easyocr import Reader
from utils import warp_image, show_image
"""
img = cv2.imread("video_test.png")

warped_img = img
#warped_img = warp_image(img, np.array(eval(str([(7, 592), (1478, 378),(1557, 1232), (224, 1526)])), dtype="float32"))
#(179, 979), (1262, 678),(1324, 1109), (350, 1522)

reader = Reader(["en"])
results = reader.readtext(warped_img)

for (bbox, text, prob) in results:
    # display the OCR'd text and associated probability
    print("[INFO] {:.4f}: {}".format(prob, text))
    print(bbox)
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    # cleanup the text and draw the box surrounding the text along
    # with the OCR'd text itself
    # text = cleanup_text(text)
    cv2.rectangle(warped_img, tl, br, (0, 255, 0), 2)
    cv2.putText(warped_img, text, (tl[0], tl[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

show_image(warped_img)
ff = 0


