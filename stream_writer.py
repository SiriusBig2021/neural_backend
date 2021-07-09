from utils import *

cameras = {
    # "bot1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:554",
    # "bot2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:555",
    "mid1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:556",
    "mid2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:557",
    "top": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:558"
}

readers = {}

writers = {}

# init readers
for cam in cameras:

    r = Reader(name=cam,
               src=cameras[cam],
               type="rtsp_stream",
               save_to_file=False,
               save_dir="./data/archive")

    readers[cam] = r

time.sleep(2)

# show frames
while True:

    for cam in readers:

        reader_out = readers[cam].get_frame()

        if "error" in reader_out:
            print(get_format_date(), "##", cam, "##", "Error", reader_out["error"])

            for camm in readers:
                #print(f'after| cam - {readers[cam].name}, {readers[cam].connected}')
                readers[camm].reset()
                #print(f'before| cam - {readers[cam].name}, {readers[cam].connected}')

            time.sleep(2)
            break

        frame, frame_metadata = reader_out["frame"], reader_out["frame_meta"]
        show_image(frame, win_name=cam, delay=1)
        # print(frame)

    # cv2.destroyAllWindows()
