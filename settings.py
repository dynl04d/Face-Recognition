"""
All GLOBAL variables will be defined here.
Whenever they are required, this file should be imported.
Note: All directory strings are referenced from the 'ImageRecognition' directory.
"""

import os


WINDOW_NAME = "camera-preview"  # Identifier for the named window for video capture

DOWNSAMPLE = 4  # Factor of which the images will be down-sampled

USEPI = False  # indicate whether or not we are on the Pi

LOCK_TIMEOUT = 10  # time to wait before closing the lock

UNKNOWN_USER = "UNKNOWN"  # the name which will be used for unclassified users

FACE_THRESHHOLD = 120.0  # distance threshold set for identifying a user

MAX_DETECTION_SEQUENCE = 11  # number of faces that should be used to determine if a user is authorised or not

DISP_FEED = True  # Indicates whether or not the video feed will be displayed


# ################## File Paths ################## #
LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + "/logs/"

PI_PREFIX_IM = os.path.dirname(os.path.abspath(__file__))

FACE_CASCADE_FILE = PI_PREFIX_IM + "res/classifiers/haarcascade_frontalface_default.xml" # XML containing face cascades

EYE_CASCADE_FILE = PI_PREFIX_IM + "res/classifiers/haarcascade_eye.xml"  # XML containing eye cascades

KEY_FACES = PI_PREFIX_IM + "res/key_faces"  # directory containing identifiers for faces

ACCESS_FILE = LOG_DIR + "admin.json"  # path to the file which contains the authorized user details

LOG_FILE = LOG_DIR + "access_log.json"
