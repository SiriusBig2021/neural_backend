import time
import time as tm
import traceback
from DenseOpticalFlow import DenseOpticalFlow
from models import OCRReader, FENN, FB_send
from utils import *
from Firebase import *
import torch

##########--config params--#################################################################
cameras = {

    # "bot1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:554",
    # "bot2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:555",
    # "mid1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:556",
    # "mid2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:557",
    # "top": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:558"

    "mid1": "./data/backend_processor_tests/mid_test_main.mp4",
    "top": "./data/backend_processor_tests/top_test_main.mp4"

    # "mid1": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo.mp4",
    # "top": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo1.mp4"

    # "mid1": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo2.mp4",
    # "top": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo3.mp4"

    # "mid1": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo4.mp4",
    # "top": "/home/ea/projects/SIRIUS21/data/backend_processor_tests/cutVideo5.mp4"

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

nn_cfg = {

    "device": "cpu",  # "cpu" or "cuda:0" for gpu
    "input_shape": (3, 128, 128),  # ch, h, w
    "classes": ['empty', 'fill'],
    "pathToWeights": "./fill_classifier.pt"

}

max_wait_iteration = 4
cut_cord_mid1 = [(0, 249), (1296, 249), (1296, 1065), (0, 1065)]
do_imshow = False
do_save_results = True
############################################################################################

##########--initialization--################################################################
ocr = OCRReader(type="rtsp", gpu=False)
op = DenseOpticalFlow(opt_param)

model = FENN(input_shape=nn_cfg["input_shape"], classes=nn_cfg["classes"], deviceType=nn_cfg["device"])
model.load_state_dict(torch.load(nn_cfg["pathToWeights"]))

print(os.getpid())
firebase = FB_send()
print(2)
# DC = DataComposer()
# DC.CreateCurrentShift()  # TODO необходимо создавать вначале смены + trainID
############################################################################################

##########--text decoration--###############################################################
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 255, 0)
thickness = 2
#rr = cv2.namedWindow("result")
wk = 0
############################################################################################
if __name__ == "__main__":
    os.setpgrp()
    try:
        readers = {cam: Reader(name=cam, src=cameras[cam], type="file") for cam in cameras}
        tm.sleep(2)
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
                        tm.sleep(2)
                        print(c, " - have been reseated successfully")
                    break
                frame = reader_out["frame"]
                moment_frames[camera] = {}
                moment_frames[camera]["frame"] = frame
            if "mid1" and "top" not in moment_frames:
                continue
            else:
                cut_frame_mid1 = warp_image(moment_frames["mid1"]["frame"], np.array(eval(str(cut_cord_mid1)), dtype="float32"))

                first_t = tm.time()
                # movement_direct = "left"
                movement_direct = op.getMoveDirection(cut_frame_mid1)
                second_t = tm.time()
                print("\r", second_t - first_t, f"direction {movement_direct}", end="")
                if movement_direct != "wait" and "up" and "down":
                    ocr_handler = ocr.main_ocr_run(cut_frame_mid1, max_wait_iteration)

                    if ("flag" in ocr_handler) and (len(top_buf) == 0):
                        moment_frames["top"]["direction"] = movement_direct
                        top_buf = moment_frames["top"]
                        top_buf["time"] = get_format_date(date_format="%Y-%m-%dT%H:%M:%S")
                        print(top_buf["time"])
                        # print(len(top_buf))
                        print("wagon with number, and it goes to the buffer")

                    elif "prob" in ocr_handler:
                        predict = model.predict(top_buf["frame"])
                        predict_class, predict_prob = predict["className"], predict["accuracy"],
                        top_buf["state"] = predict_class
                        #################--Drawing_bbox--#################################
                        text = ocr_handler["number"]
                        cv2.putText(ocr_handler["frame"], movement_direct, (30, 40), fontFace, fontScale, color, thickness)
                        cv2.putText(top_buf["frame"], top_buf["time"], (30, 40), fontFace, fontScale, color, thickness)
                        ##################################################################
                        ################--Save_image--####################################
                        #TODO-----------------------------------------------------------------------------------
                        # cv2.imwrite(f"./data/results_of_backend/{top_buf['time']} - mid-MAIN.jpg", cut_frame_mid1)
                        # show_image(ocr_handler["frame"], win_name="mainMain", delay=1)
                        #TODO-----------------------------------------------------------------------------------
                        if do_save_results:
                            cv2.imwrite(f"./data/results_of_backend/{top_buf['time']} - mid1.jpg", ocr_handler["frame"])
                            cv2.imwrite(f"./data/results_of_backend/{top_buf['time']} - top.jpg", top_buf["frame"])
                        ##################################################################
                        counter += 1
                        all_info[counter] = {"top": {"frame": top_buf["frame"],
                                                     "time": top_buf["time"],
                                                     "state": top_buf["state"]},
                                             "mid": {"frame": ocr_handler["frame"],
                                                     "number": ocr_handler["number"]}}

                        event = {
                            "time": top_buf["time"],
                            "direction": "arrive" if top_buf["direction"] == "left" else "departure",
                            "number": text,
                            "trainID": 4,
                            "state": top_buf["state"],
                            "event_frames": [
                                {
                                    'camera': 'top',
                                    'imagePath': f"./data/results_of_backend/{top_buf['time']} - top.jpg"
                                },
                                {
                                    'camera': 'mid',
                                    'imagePath': f"./data/results_of_backend/{top_buf['time']} - mid1.jpg"
                                }]
                        }
                        firebase.send_to_process(event)
                        #####################################################################
                        #st = tm.time()
                        #DC.AddEvent(event["time"],
                        #            event["direction"],
                        #            event["number"],
                        #            event["trainID"],
                        #            event["state"],
                        #            event["event_frames"]
                        #            )
                        #print("firebase time", tm.time() - st)
                        #top_buf.clear()
                        ###################################################################

            # if len(all_info) == 0:
            #     print("\r", "нет значений", end="")
            # else:
            #     print(all_info)

            ############################--Images showing--#############################################################
            if do_imshow:
                cv2.putText(cut_frame_mid1, movement_direct, (30, 40), fontFace, fontScale, color, thickness)
                # show_image(cut_frame_mid1, win_name="mid1_circumcised", delay=1)
                cv2.imshow("result", cut_frame_mid1)
                key = cv2.waitKey(wk) & 0xff
                if key == ord('p'):
                    if wk == 0:
                        wk = 1
                    elif wk == 1:
                        wk = 0
                for i in moment_frames:
                    show_image(moment_frames[i]["frame"], win_name=i, delay=1)
            ##########################################################################################################
            moment_frames.clear()

    except:
        print(traceback.format_exc())

    finally:
        os.killpg(0, signal.SIGKILL)
