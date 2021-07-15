import time
import traceback

from utils import *

do_imshow = True
write_archive = True
archive_dir = "./data/archive"
chunk_max_frames = 50000
write_to_new_file_if_reconnect = False

cameras = {
    # "bot1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:554",
    # "bot2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:555",
    "mid1": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:556",
    # "mid2": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:557",
    # "top": "rtsp://user:bDC8BzQeFp8jb0C@217.195.100.69:558"
}

if __name__ == "__main__":
    os.setpgrp()

    try:

        # init reader process for each camera
        readers = {cam: Reader(name=cam, src=cameras[cam], type="rtsp_stream") for cam in cameras}
        time.sleep(2)

        writers = {cam: None for cam in cameras}
        recorded_frames = 0
        recorded_chunks = 0

        while True:

            log_str = ""
            for cam in readers:

                reader_out = readers[cam].get_frame()

                if "error" in reader_out:
                    print(get_format_date(), "##", cam, "##", "Error", reader_out["error"])

                    # reconnect to all cams if one disconnected
                    for c in readers:
                        readers[c].reset()
                        if write_archive and writers[c] is not None and write_to_new_file_if_reconnect:
                            writers[c].finish_writing()
                    if write_to_new_file_if_reconnect:
                        cv2.destroyAllWindows()
                    time.sleep(2)
                    break

                log_str += f"{cam}: {reader_out['time']} "

                frame, metadata = reader_out["frame"], reader_out["meta"]

                if do_imshow:
                    show_image(frame, win_name=cam, delay=1)

                if write_archive:
                    if not os.path.isdir(archive_dir):
                        os.makedirs(archive_dir)

                    # init writers
                    if writers[cam] is None or recorded_frames % chunk_max_frames == 0:
                        recorded_chunks += 1

                        # finish writing if chunk filled
                        if writers[cam] is not None:
                            writers[cam].finish_writing()

                        save_file = os.path.join(archive_dir, f"{cam}_{get_format_date()}.mp4")

                        writers[cam] = Writer(file_name=save_file,
                                              fps=metadata["fps"],
                                              height=metadata["h"],
                                              width=metadata["w"])

                    writers[cam].write_to_file(frame)
                    recorded_frames += 1

            if write_archive:
                log_str += f"writing to {archive_dir}"

            print(get_format_date(), f"{log_str}")

    except:
        print(traceback.format_exc())

    finally:
        os.killpg(0, signal.SIGKILL)
