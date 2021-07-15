import numpy as np
import cv2
import os
import pickle


def get_filelist(directory, ext, separate=False):
    """
    get files list with required extensions

    separate=True creates [[path], [name], [ext]]
    that may be useful is case of work with images/labels files

    Usage:
    imgs_list = get_filelist(dir, ".jpg", True)
    labels_list = get_filelist(dir, ".txt", True)
    for label_name in labels_list[1]:
        id = labels_list[1].index(label_name)
        label_file = os.path.join(labels_list[0][id], label_name + ".txt")
        img_file = os.path.join(imgs_list[0][id], label_name + ".jpg")
    ...
    """

    ret_list = []
    if separate:
        ret_list = [[], [], []]
    for folder, subs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ext):
                if separate:
                    ret_list[0].append(folder)
                    ret_list[1].append(".".join(filename.split(".")[:-1]))
                    ret_list[2].append(ext)
                else:
                    ret_list.append(os.path.join(folder, filename))
    return ret_list


def resize_image(img, height=None, width=None, letterbox=False, lb_color=(128, 128, 128), inter=cv2.INTER_AREA):

    """
    img        -  np.array
    height     -  new H
    width      -  new W
    letterbox  -  add areas to lowest side for escaping image distortion
    lb_color   -  color of areas, default is gray
    """

    if letterbox:
        h, w = img.shape[:2]
        if h > w:
            border_y = 0
            border_x = round((h - w + .5) / 2.)
        else:
            border_x = 0
            border_y = round((w - h + .5) / 2.)
        img = cv2.copyMakeBorder(img, top=border_y, bottom=border_y, left=border_x, right=border_x,
                                 borderType=cv2.BORDER_CONSTANT, value=lb_color)

    dim = (width, height)
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    if height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(img, dim, interpolation=inter)


def mouse_callb_points(event, x, y, r1, r2):
    global corners
    global draw_mode

    if draw_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append(np.array([float(x), float(y)], dtype=np.float32))
            objpoints.append(np.array([float(x), float(y), float(0)], dtype=np.float32))
            print(corners)

        elif event == cv2.EVENT_LBUTTONUP:
            cv2.circle(img, tuple(int(x) for x in corners[-1]), 2, (10, 10, 255), 2)
            # if len(corners) > 1:
            #     cv2.line(img, tuple(corners[-2][0]), tuple(corners[-1][0]), (0, 0, 255), 2)

            cv2.imshow("image", img)


def calibrate(objectPointsArray, imgPointsArray, shape):
    # print(objectPointsArray)
    # objectPointsArray = [
    #     [0., 100., 0.],
    #     # [200., 100.,   0.],
    #     [400., 100., 0.],
    #     # [600., 100.,   0.],
    #     [800., 100., 0.],
    #     [0., 0., 0.],
    #     # [200., 0.,   0.],
    #     [400., 0., 0.],
    #     # [600., 0.,   0.],
    #     [800., 0., 0.]]
    # # objectPointsArray = objpoints
    # objectPointsArray = [np.array(objectPointsArray, dtype=np.float32)]
    # print(objectPointsArray)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, shape, None, None)
    return mtx, dist


def undistort(imgs_list, mtx, dist):

    for fname in imgs_list:
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        # 1
        dst = cv2.undistort(img, mtx, dist, None, None)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # # 2
        # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]
        cv2.namedWindow("undistorted", cv2.WINDOW_NORMAL)
        cv2.imshow("undistorted", dst)
        cv2.waitKey()


if __name__ == "__main__":

    # images_dir = "/home/ea/projects/lom/distrtion_chests"
    images_dir = "/home/ea/projects/lom/distortion_one_wagon"

    use_camera_calibration = False

    draw_points = True
    use_ready_points = False

    rows = 4
    cols = 2

    # -------------------------------------------------------------------------------

    imgs_list = sorted(get_filelist(images_dir, ext=".jpg"))
    mtx = None
    dist = None

    if use_camera_calibration:

        points_file = "./{}.pkl".format(images_dir.split("/")[-1])
        if os.path.isfile(points_file):
            with open(points_file, "rb") as ff:
                ready_points = pickle.load(ff)

        corners = []
        objpoints = []
        draw_mode = False

        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

        objectPoints = np.zeros((rows * cols, 3), np.float32)
        objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

        objectPointsArray = []
        imgPointsArray = []

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)

        if use_ready_points and os.path.isfile(points_file):
            objectPointsArray = ready_points[0]
            imgPointsArray = ready_points[1]

            for n, fname in enumerate(imgs_list):
                if n > 0: break
                img = cv2.imread(fname)
                img = resize_image(img, width=800)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for pts in imgPointsArray[n]:
                    cv2.circle(img, tuple(int(x) for x in pts), 2, (10, 10, 255), 2)

                cv2.imshow("image", img)
                cv2.waitKey()

        else:
            for fname in imgs_list[:1]:

                img = cv2.imread(fname)
                img = resize_image(img, width=800)

                if draw_points:
                    cv2.imshow("image", img)
                    key = cv2.waitKey(0) & 0xff
                    if key == ord('q'):
                        if draw_mode is False:
                            draw_mode = True
                            print("draw mode ON")
                            while draw_mode is True:
                                cv2.imshow("image", img)
                                cv2.setMouseCallback("image", mouse_callb_points)
                                key = cv2.waitKey(0) & 0xff
                                if key == ord('q'):
                                    print("draw mode OFF")
                                    draw_mode = False

                    corners = np.array(corners)
                    print(corners)
                    objectPointsArray.append(objectPoints)
                    imgPointsArray.append(corners)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # TODO
                    # corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    # cv2.drawChessboardCorners(img, (rows, cols), corners, True)
                    # cv2.imshow("image", img)
                    # print("ok")

                    corners = []

                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Find the chess board corners
                    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
                    # If found, add object points, image points (after refining them)
                    print(corners)

                    if ret is True:
                        objectPointsArray.append(objectPoints)

                        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        imgPointsArray.append(corners)

                        cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
                        cv2.imshow("image", img)
                        cv2.waitKey()

            if not len(objectPoints) or not len(imgPointsArray):
                print("Can`t find objects points. Exit")
                exit()

            if not os.path.isfile(points_file):
                with open(points_file, "wb") as ff:
                    pickle.dump([objectPointsArray, imgPointsArray], ff)

        mtx, dist = calibrate(objectPointsArray, imgPointsArray, gray.shape[::-1])

    if mtx is None or dist is None:
        print("Exit")
        exit()
        
    undistort(imgs_list, mtx, dist)
