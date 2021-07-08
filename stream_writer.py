from utils import *

cameras = {
    "bot1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:554",
    "bot2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:555",
    "mid1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:556",
    "mid2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:557",
    "top": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:558"
}

readers = {}
writers = {}

# init readers
for cam in cameras:

    r = Reader(name=cam, src=cameras[cam], type="rtsp_stream",
               save_to_file=True, save_file="./data/archive/%s.mp4" % cam)

    readers[cam] = {"status": False, "r": r, "frame": None}

while True:

    i_can_write = False

    for cam in readers:

        reader_out = readers[cam]["r"].get_frame()

        if "error" in reader_out:
            print(get_format_date(), "##", cam, "##", "Error", reader_out["error"])
            status = {"error": reader_out["error"]}
            readers[cam]["status"] = False
            continue

        readers[cam]["status"] = True
        readers[cam]["frame"] = reader_out["frame"]

    cams_status = {readers[x]["status"] for x in readers}

    if False in cams_status:
        continue

    frame, frame_metadata = reader_out["frame"], reader_out["frame_meta"]
    show_image(frame, win_name=cam, delay=1)
