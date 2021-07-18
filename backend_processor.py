import time
import traceback
from DenseOpticalFlow import DenseOpticalFlow
from models import OCRReader
from utils import *

##########--config params--#################################################################
# "bot1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:554",
# "bot2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:555",
#"mid1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:556",
# "mid2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:557",
#"top": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:558"

cameras = {
    "mid1": "/home/home/projects/neural_backend/data/backend_processor_tests/mid_test_main.mp4",
    "top": "/home/home/projects/neural_backend/data/backend_processor_tests/top_test_main.mp4"
}
opt_param = {
    'threshold_magnitude': 9,
    'size_accumulation': 6,
    'opticflow_param': {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 100,
        'iterations': 4,
        'poly_n': 5,
        'poly_sigma': 1.1,
        'flags': cv2.OPTFLOW_LK_GET_MIN_EIGENVALS
    }
}
max_wait_iteration = 4
cut_cord_mid1 = [(288, 302), (1209, 302), (1209, 1041), (288, 1041)]
do_imshow = True
############################################################################################

##########--initialization--################################################################
ocr = OCRReader(type="rtsp")
op = DenseOpticalFlow(opt_param)
FENN = None
############################################################################################

##########--text decoration--###############################################################
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 255, 0)
thickness = 2
############################################################################################
if __name__ == "__main__":
    os.setpgrp()
    try:
        readers = {cam: Reader(name=cam, src=cameras[cam], type="file") for cam in cameras}
        time.sleep(2)
        top_buf = {}
        all_info = {}
        counter = 0

        while True:
            moment_frames = {}
            for camera in readers:
                reader_out = readers[camera].get_frame()

                if "error" in reader_out:
                    print(get_format_date(), "##", camera, "##", "Error", reader_out["error"])
                    for c in readers:
                        readers[c].reset()
                        time.sleep(2)
                        print(c, " - have been reseated successfully")
                    break
                frame = reader_out["frame"]
                moment_frames[camera] = {}
                moment_frames[camera]["frame"] = frame
            if "mid1" and "top" not in moment_frames:
                continue
            else:
                cut_frame_mid1 = warp_image(moment_frames["mid1"]["frame"], np.array(eval(str(cut_cord_mid1)), dtype="float32"))

                # is_movement = NeuralModel(frame)
                # if not is_movement:
                #     continue

                first_t = time.time()
                movement_direct = op.getMoveDirection(cut_frame_mid1)
                second_t = time.time()
                print("\r", second_t - first_t, f"direction {movement_direct}", end="")
                if movement_direct != "wait" and "up" and "down":
                    ocr_handler = ocr.main_ocr_run(cut_frame_mid1, max_wait_iteration)

                    print("\n", ocr_handler)
                    print("empty frames - ", ocr.empty_frames)

                    if ("flag" in ocr_handler) and (len(top_buf) == 0):
                        moment_frames["top"]["direction"] = movement_direct
                        top_buf = moment_frames["top"]
                        top_buf["time"] = time.ctime(time.time())
                        print(len(top_buf))
                        print("wagon with number, and it goes to the buffer")

                    elif "prob" in ocr_handler:
                        # fn_prob = FENN.get_prediction(top_buf["frame"]).getClassName()
                        # top_buf["state"] = fn_prob
                        #################--Drawing bbox--#################################
                        tl = ocr_handler["bbox"][1]
                        br = ocr_handler["bbox"][2]
                        text = ocr_handler["number"]
                        cv2.rectangle(cut_frame_mid1, tl, br, (0, 255, 0), 2)
                        cv2.putText(cut_frame_mid1, text, (tl[0], tl[1] - 10), fontFace, fontScale, color, thickness)
                        cv2.putText(cut_frame_mid1, text, (tl[0], tl[1] - 10), fontFace, fontScale, color, thickness)
                        ##################################################################
                        counter += 1
                        all_info[counter] = {"top": {"frame": top_buf["frame"],
                                                     "time": top_buf["time"],
                                                     "state": "top_buf['state']"},
                                             "mid": {"frame": ocr_handler["frame"],
                                                     "number": ocr_handler["number"]}}
                        top_buf.clear()
                        # print(f"номер вагона = {ocr_handler['number']}")

            # if len(all_info) == 0:
            #     print("\r", "нет значений", end="")
            # else:
            #     print(all_info)
            #TODO записывать изображения с двух камер попарно + добавлять на изображение "номер" + "направление движения
            # "
            ############################--Images showing--#############################################################
            if do_imshow:
                cv2.putText(cut_frame_mid1, movement_direct, (30, 40), fontFace, fontScale, color, thickness)
                show_image(cut_frame_mid1, win_name="mid1_circumcised", delay=1)
                for i in moment_frames:
                    show_image(moment_frames[i]["frame"], win_name=i, delay=1)
            ##########################################################################################################
            if len(all_info) > 0:
                print(len(all_info))
            moment_frames.clear()

    except:
        print(traceback.format_exc())

    finally:
        os.killpg(0, signal.SIGKILL)
