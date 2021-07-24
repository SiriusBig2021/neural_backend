import os
import cv2
import utils
import copy
import numpy as np

os.environ[
    'GOOGLE_APPLICATION_CREDENTIALS'] = '/home/sauce-chili/Sirius/neural_backend/ApiKeys/testOCR-320608-62be95bded39.json'

from google.cloud import vision

# imgs = utils.get_filelist("./data/ocr_test", ["jpg", "png"])

save_dir = "/home/sauce-chili/Sirius/neural_backend/data/ocr_dataset"
do_save = True
pathToVideo = '/home/sauce-chili/Sirius/neural_backend/data/VideoWithMove/mid_1/mid1_11-07-2021_06:19:41_15-07-2021-11:37.mp4'
pathToImg = '/home/sauce-chili/Sirius/neural_backend/data/Img/frame_for_warp.png'

cap = cv2.VideoCapture(pathToVideo)
client = vision.ImageAnnotatorClient()

warp_rect = np.float32(
    [(658, 170), (1034, 161), (1018, 514), (682, 716)]
    # tl            tr             br         bl
)


def detect_number(frame):
    img_str = cv2.imencode('.jpg', frame)[1].tostring()
    image = vision.Image(content=img_str)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    texts = response.text_annotations

    # print(texts)
    predicts = []
    ret_data = []
    predicted_texts = []
    for text in texts:

        # print('\n"{}"'.format(text.description))
        vertices_x = []
        vertices_y = []
        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #              for vertex in text.bounding_poly.vertices])
        for vertex in text.bounding_poly.vertices:
            # cv2.circle(frame, (vertex.x, vertex.y), 3, (0, 0, 255), 2)
            vertices_x.append(vertex.x)
            vertices_y.append(vertex.y)

        text_str = text.description.replace('|', '')

        if len(text_str) != 8 or ":" in text_str or "Camera" in text_str:
            continue

        left = min(vertices_x)
        top = min(vertices_y)
        width = max(vertices_x) - min(vertices_x)
        height = max(vertices_y) - min(vertices_y)

        print('Vertices_x', end=' ')
        print(vertices_x)
        print('Vertices_y', end=' ')
        print(vertices_y)

        # cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 1)
        # utils.draw_bbox(frame, utils.xywh_to_xyxy(frame.shape[0], frame.shape[1], number["bbox"]),
        #                 label=number["name"])
        #
        # cv2.imshow('Bounding digit', frame)

        predict = {"type": "number",
                   "bbox": [left / frame.shape[1], top / frame.shape[0], width / frame.shape[1],
                            height / frame.shape[0]],
                   "prob": 1., "name": text_str}

        predicts.append(predict)

    return predicts


# frame = cv2.imread(pathToImg)

while True:

    grabbed, frame = cap.read()

    if not grabbed:
        print("[INFO] Video isn't capture")
        break

    # print(frame.shape)
    # utils.show_image(frame, win_name="origin", delay=1)

    # utils.draw_bbox(frame, [0, 240, 920, 720])
    # utils.show_image(frame, "orig", delay=1)
    warp_img = utils.warp_image(frame, warp_rect)
    # warp_img = frame
    rr = cv2.namedWindow("rr")
    cv2.imshow("rr", warp_img)
    key = cv2.waitKey(0)

    if key == ord('s'):
        predicts = detect_number(warp_img)

        img_labels = ""
        draw_img = copy.deepcopy(warp_img)
        for number in predicts:
            print(number)

            bbox = utils.xywh_to_xyxy(warp_img.shape[0], warp_img.shape[1], number["bbox"])
            # utils.draw_bbox(draw_img, bbox, label=number["name"])
            # utils.show_image(warp_img, win_name='draw bbox')
            bbox_w = bbox[2] - bbox[0]

            while '\n' in number["name"]:
                number["name"] = number["name"].replace('\n','          ')

            number_split = int(bbox_w * 0.03)
            numbers_w = {"0": 0.14,
                         "1": 0.05,
                         "2": 0.12,
                         "3": 0.11,
                         "4": 0.16,
                         "5": 0.1,
                         "6": 0.12,
                         "7": 0.12,
                         "8": 0.1,
                         "9": 0.12}
            prev_c = bbox[0]
            for n, i in enumerate(number["name"]):
                number_bbox = [prev_c, bbox[1], int(prev_c + bbox_w * numbers_w[i]), bbox[3]]
                # print("prev_c:", prev_c, "bbox:",number_bbox)

                utils.draw_bbox(draw_img, number_bbox, label=i)

                utils.show_image(draw_img, delay=1)

                number_bbox_rel = utils.xyxy_to_xcycwh(warp_img.shape[0], warp_img.shape[1], number_bbox)

                bbox_str = "{} {} {} {} {}\n".format(
                    i, number_bbox_rel[0], number_bbox_rel[1], number_bbox_rel[2], number_bbox_rel[3])

                img_labels += bbox_str

                prev_c += number_bbox[2] - number_bbox[0] + number_split

            # # number_img = copy.deepcopy(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            #
            # # cv2.imshow('Number img', number_img)
            #
            # number_img_result = detect_number(number_img)
            #
            # for number2 in number_img_result:
            #     bbox = utils.xywh_to_xyxy(number_img.shape[0], number_img.shape[1], number2["bbox"])
            #     utils.draw_bbox(number_img, bbox, label=number2["name"])
            #
            # utils.show_image(number_img)

        if do_save:
            rnd_name = utils.get_random_name()
            img_save_path = os.path.join(save_dir, "{}.jpg".format(rnd_name))
            print("saved to", os.path.join(save_dir, "{}.jpg".format(rnd_name)))
            cv2.imwrite(img_save_path, warp_img)

            label_save_path = os.path.join(save_dir, "{}.txt".format(rnd_name))
            with open(label_save_path, "w") as f:
                f.write(img_labels)

    # utils.show_image(warp_img, win_name='warped', delay=1)
    # cv2.imshow('Result', frame)
    # cv2.waitKey(0)
