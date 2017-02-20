import cv2
from threading import Thread


class Camera(object):
    """
    The camera class will interface with the device camera. Here we
    can define functions to capture the current video stream, save the
    stream, or adjust the frame rate.
    """

    def __init__(self):
        self._frame = None
        self._videoFeed = cv2.VideoCapture(0)
        self._videoFeed.set(5, 10)  # set to 10 FPS
        (self._grabbed, self._frame) = self._videoFeed.read()
        self._stopped = False

    def __del__(self):
        self._videoFeed.release()

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self._stopped:
                break

            # otherwise, read the next frame from the stream
            (self._grabbed, self._frame) = self._videoFeed.read()
        return

    def read(self):
        # return the frame most recently read
        return self._frame

    def stop(self):
        # indicate that the thread should be stopped
        self._stopped = True
        return


class PiCamera(object):
    """
    The camera class will interface with the raspberry pi camera.
    If the code is to be run on the RPi, then the __init__ function
    should have its code uncommented. On the Pi the picamera module
    should be available.
    """

    def __init__(self):
        # import the necessary packages
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        self._camera = PiCamera()
        self._camera.resolution = (640, 480)
        self._raw_capture = PiRGBArray(self._camera, size=(640, 480))
        self._frame = None
        self._stopped = False

    def __del__(self):
        self._raw_capture.truncate()

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # capture frames from the camera
        for frame in self._camera.capture_continuous(self._raw_capture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            image = frame.array

            self._frame = image

            # clear the stream in preparation for the next frame
            self._raw_capture.truncate(0)

            if self._stopped:
                return
        return

    def read(self):
        # return the frame most recently read
        return self._frame

    def stop(self):
        # indicate that the thread should be stopped
        self._stopped = True
        return
