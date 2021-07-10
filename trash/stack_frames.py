from utils import *
import cv2
from panorama import Panaroma

cap = cv2.VideoCapture("./data/archive/top_4.mp4")

class Undistorter:

    def __init__(self):

        self.h = 240
        self.w = 1060

        self.distortion_cfg = {
            "cam1": {

                "mtx": np.array([[22426.52646, 100.0, 2376.8923999999997],
                                 [601.0, 7532.283019999999, 1310.65076],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float64),

                "dist": np.array([[-1.90263172, -10.15574996, -0.22993206, -0.00856858, -0.007612]], dtype=np.float64),

                "pts1": np.float32([tuple([178, 118]), tuple([1278, 153]), tuple([10, 726]), tuple([1278, 514])]),
                "pts2": np.float32([[0, 0], [self.w, 0], [0, self.h], [self.w, self.h]])
            }
        }

    def undistort(self, frame, cam_name):

        dst = cv2.undistort(src=frame,
                            cameraMatrix=self.distortion_cfg[cam_name]["mtx"],
                            distCoeffs=self.distortion_cfg[cam_name]["dist"],
                            dst=None,
                            newCameraMatrix=None)

        M = cv2.getPerspectiveTransform(src=self.distortion_cfg[cam_name]["pts1"],
                                        dst=self.distortion_cfg[cam_name]["pts2"])

        dst = cv2.warpPerspective(dst, M, (self.w, self.h))
        #
        # if cam_name == "cam2":
        #     dst = np.delete(dst, np.s_[:int(frame.shape[1] * 0.02)], axis=1)
            # dst = resize_image(dst, undistorter.h, undistorter.w)

        return dst


    def correct_params(self, frame, cam_name):

        im = cv2.namedWindow("image")

        p = {

            "arr": None,
            "c": None,
            "c1": None,

            "mtx": [[100, 100, 100], [100, 100, 100], [1, 0.1, 0.1]],
            "dist": [0.1, 1, 0.01, 0.05, 10]

        }

        while True:

            # print(self.distortion_cfg[cam_name]["mtx"])
            img = self.undistort(frame, cam_name)

            if p["arr"] == "mtx":
                print("\rCACHED ARR {} CACHED VALUE INDEX [{}][{}] VALUE {}".format(
                    p["arr"], p["c1"], p["c"], self.distortion_cfg[cam_name]["mtx"][p["c1"]][p["c"]]), end="")
            if p["arr"] == "dist":
                print("\rCACHED ARR {} CACHED VALUE INDEX [{}] VALUE {}".format(
                    p["arr"], p["c"], self.distortion_cfg[cam_name]["dist"][0][p["c"]]), end="")

            cv2.imshow("image", img)
            key = cv2.waitKey(0) & 0xff

            if key == ord("x"):
                if p["arr"] == "mtx":
                    self.distortion_cfg[cam_name]["mtx"][p["c1"]][p["c"]] += p["mtx"][p["c1"]][p["c"]]
                if p["arr"] == "dist":
                    self.distortion_cfg[cam_name]["dist"][0][p["c"]] += p["dist"][p["c"]]

            if key == ord("z"):
                if p["arr"] == "mtx":
                    self.distortion_cfg[cam_name]["mtx"][p["c1"]][p["c"]] -= p["mtx"][p["c1"]][p["c"]]
                if p["arr"] == "dist":
                    self.distortion_cfg[cam_name]["dist"][0][p["c"]] -= p["dist"][p["c"]]

            if chr(key) in ["0","1","2"]:
                p["arr"] = "mtx"
                p["c"] = ["0","1","2"].index(chr(key))
                p["c1"] = 0

            if chr(key) in ["3", "4", "5"]:
                p["arr"] = "mtx"
                p["c"] = ["3", "4", "5"].index(chr(key))
                p["c1"] = 1

            if chr(key) in ["6", "7", "8"]:
                p["arr"] = "mtx"
                p["c"] = ["6", "7", "8"].index(chr(key))
                p["c1"] = 2

            if chr(key) in ["q", "w", "e", "r", "t"]:
                p["arr"] = "dist"
                p["c"] = ["q", "w", "e", "r", "t"].index(chr(key))


undistorter = Undistorter()
panaroma = Panaroma()

wagon_frames = []
while True:

    _, frame = cap.read()

    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print("\rFRAME:", frame_id, end="")

    if frame_id < 451:
        continue

    if frame_id == 520:
        break

    wagon_frames.append(frame)
    # show_image(frame, delay=1)

stacked_img = None

for n, _ in enumerate(wagon_frames):
    if n + 1 == len(wagon_frames):
        break

    if n % 10 != 0:
        continue

    img1 = wagon_frames[n]
    img2 = wagon_frames[n+1]

    undistorter.correct_params(img1, "cam1")

    # img1 = undistorter.undistort(img1, "cam1")
    # img2 = undistorter.undistort(img2, "cam1")

    # img1 = resize_image(img1, width=800)
    # img2 = resize_image(img2, width=800)

    if stacked_img is None:
        # stacked_img = np.hstack((img1, img2))
        stacked_img = img1
    else:
        stacked_img = np.hstack((stacked_img, img2))

    # (stacked_img, matched_points) = panaroma.image_stitch([img1, img2], match_status=True)

        show_image(stacked_img)