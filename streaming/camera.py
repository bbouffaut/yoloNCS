import cv2
import threading
import time

class VideoCamera(threading.Thread):

    def __init__(self):
        """
        Initialize video.

        """
        threading.Thread.__init__(self)
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.start()

    def run(self):
        """
        Runs a thread.

        """
        #implement a video frame buffering thread
        self.lock = threading.Lock()

        while True:
            self.lock.acquire()
            success, self.image = self.video.read()
            self.lock.release()
            time.sleep(0.1)

    def __del__(self):
        """
        Release video.

        """
        self.video.release()

    def get_frame_jpg(self):
        """
        Extracts the jpegimage from the image.

        """
        self.lock.acquire()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        image = self.image
        self.lock.release()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def get_frame_cv2_format(self):
        """
        Get image format

        """
        self.lock.acquire()
        image = self.image
        self.lock.release()

        return image
