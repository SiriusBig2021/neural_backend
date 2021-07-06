from utils import *

cameras = {
    "bot1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:554",
    "bot2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:555",
    "mid1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:556",
    "mid2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:557",
    "top": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:558"
}

readers = {}

# init readers
for cam in cameras:

    r = Reader(name=cam, src=cameras[cam], type="rtsp_stream",
               save_to_file=True, save_file="./data/archive/%s.mp4" % cam)

    readers[cam] = r


# show frames
while True:

    for cam in readers:

        reader_out = readers[cam].get_frame()

        if "error" in reader_out:
            print(get_format_date(), "##", cam, "##", "Error", reader_out["error"])
            status = {"error": reader_out["error"]}
            continue

        if reader_out["frame"] is None:
            print(get_format_date(), "##", cam, "##", "None frame")
            continue

        frame, frame_metadata = reader_out["frame"], reader_out["frame_meta"]
        show_image(frame, win_name=cam, delay=1)
