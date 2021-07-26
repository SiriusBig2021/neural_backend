import time as tm
from DenseOpticalFlow import DenseOpticalFlow
from models_NN import OCRReader, FENN, FB_send, Config, time_zone
from utils import *
import torch

##########--config params--#################################################################
############################################################################################
# """
cfg = Config('config.yaml')
cameras = cfg.cameras
opt_param = cfg.optical_params
NN_full_empty_cfg = cfg.fenn_all_fe
NN_train_cfg = cfg.fenn_all_tr
ocr_type = cfg.type_ocr
ocr_gpu = cfg.gpu_ocr
source = cfg.src
max_wait_iteration = cfg.max_wait_iteration
cut_cord_mid1 = cfg.cut_cord
do_imshow = cfg.image_show
do_save_results = cfg.saving_results
dir_for_save = cfg.dir_for_save
flag_4img = cfg.flag_4img
fontFace = cfg.fontFace
fontScale = cfg.fontScale
color = cfg.color
thickness = cfg.thickness
plus_tm = cfg.time_zone
nn_type = cfg.nn_type
# """
############################################################################################
##########--initialization--################################################################
ocr = OCRReader(type=ocr_type, gpu=ocr_gpu, nn=nn_type)   #TODO [][][][][][][][][][][][][][][][][][][][]
op = DenseOpticalFlow(opt_param)

model1 = FENN(input_shape=NN_full_empty_cfg["input_shape"], classes=NN_full_empty_cfg["classes"], deviceType=NN_full_empty_cfg["device"])
model1.load_state_dict(torch.load(NN_full_empty_cfg["pathToWeights"]))
model2 = FENN(input_shape=NN_train_cfg["input_shape"], classes=NN_train_cfg["classes"], deviceType=NN_train_cfg["device"])
model2.load_state_dict(torch.load(NN_train_cfg["pathToWeights"]))

firebase = FB_send()
############################################################################################

if __name__ == "__main__":
    os.setpgrp()
    try:
        readers = {cam: Reader(name=cam, src=cameras[cam], type=source) for cam in cameras}
        tm.sleep(2)
        top_buf = {}
        train_here = False
        train_id = 0
        counter_tumb = False

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
                now_time = tm.localtime()
                if (now_time.tm_hour == 7 and now_time.tm_min == 30) or (now_time.tm_hour == 19 and now_time.tm_min == 30):
                    train_id = 0
                predict2 = model2.predict(moment_frames["mid1"]["frame"])
                predict_class_2, predict_prob_2 = predict2["className"], predict2["accuracy"],
                if predict_class_2 != 'Train':
                    train_here = False
                    counter_tumb = False
                    continue
                train_here = True
                if counter_tumb == False:
                    train_id += 1
                    counter_tumb = True
                # TODO-----------------------------------------------------------------
                cut_frame_mid1 = warp_image(moment_frames["mid1"]["frame"], np.array(eval(str(cut_cord_mid1)), dtype="float32"))
                first_t = tm.time()
                movement_direct = op.getMoveDirection(cut_frame_mid1)
                second_t = tm.time()

                print("\r", second_t - first_t, f"direction {movement_direct}", end="")

                if movement_direct != "wait" and "up" and "down":
                    ocr_handler = ocr.main_ocr_run(cut_frame_mid1, max_wait_iteration)

                    if ("flag" in ocr_handler) and (len(top_buf) == 0):
                        moment_frames["top"]["direction"] = movement_direct
                        top_buf = moment_frames["top"]
                        top_buf["time"] = time_zone(tm=plus_tm)

                        print("time from the top", top_buf["time"])
                        print("wagon with number, and it goes to the buffer")

                    elif "prob" in ocr_handler:
                        predict1 = model1.predict(top_buf["frame"])
                        predict_class_1, predict_prob_1 = predict1["className"], predict1["accuracy"],
                        top_buf["state"] = predict_class_1

                        #################--Drawing_bbox--#################################
                        text = ocr_handler["number"]
                        cv2.putText(ocr_handler["frame"], movement_direct, (30, 200), fontFace, fontScale, color, thickness)
                        cv2.putText(top_buf["frame"], top_buf["time"], (30, 40), fontFace, fontScale, color, thickness)
                        ##################################################################

                        ################--Save_image--####################################
                        if do_save_results:
                            cv2.imwrite(f"{dir_for_save}{top_buf['time']} - mid1.jpg", ocr_handler["frame"])
                            cv2.imwrite(f"{dir_for_save}{top_buf['time']} - top.jpg", top_buf["frame"])
                        ##################################################################

                        event = {
                            "time": top_buf["time"],
                            "direction": "arrive" if top_buf["direction"] == "left" else "departure",
                            "number": text,
                            "trainID": train_id,
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
            ############################--Images showing--#############################################################
            if do_imshow:
                cv2.putText(cut_frame_mid1, movement_direct, (30, 40), fontFace, fontScale, color, thickness)
                # show_image(cut_frame_mid1, win_name="mid1_circumcised", delay=1)
                cv2.imshow("result", cut_frame_mid1)
                key = cv2.waitKey(flag_4img) & 0xff
                if key == ord('p'):
                    if flag_4img == 0:
                        flag_4img = 1
                    elif flag_4img == 1:
                        flag_4img = 0
                for i in moment_frames:
                    show_image(moment_frames[i]["frame"], win_name=i, delay=1)
            ##########################################################################################################
            moment_frames.clear()

    except:
        print(traceback.format_exc())

    finally:
        os.killpg(0, signal.SIGKILL)
