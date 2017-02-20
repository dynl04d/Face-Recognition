import cv2
import settings
import AccessControl
from CameraModule import Camera, PiCamera
from DetectionModule import FaceDetector, FaceIdentifier
import time
import hashlib
import json
import os


def admin_file_status(fhash=None):
    """
    Gets the MD5 hash of the user access file and checks if the contents have changed
    by comparing the hash to the previous MD5 of the same file.
    :param fhash: Previous MD5.
    :return:
    """
    curr_hash = hashlib.md5(open(settings.ACCESS_FILE, 'rb').read()).hexdigest()
    if fhash != None:
        changed = fhash != curr_hash
    else:
        changed = False
    return curr_hash, changed


def get_user_list():
    """
    Reads the user access file to generate a list of authorized users.
    This list will be used for training the face identifier.
    :return:
    """
    usr_list = []
    crop_list = []
    retry = False
    try:
        with open(settings.ACCESS_FILE, 'rb') as data_file:
            admin_data = json.load(data_file)
            data_file.close()

        for user_info in admin_data["accessDetail"]:
            usr_list.append(str(user_info["username"]))
            if not user_info["cropped"]:
                crop_list.append(str(user_info["username"]))

    except:
        retry = True
    usr_list.append("undefined")  # add undefined for the case of a single user (SVM needs to compare at least 2 items)
    return usr_list, crop_list, retry


def update_access_list(preprocess_lst):
    """
    Updates the 'cropped' attribute to True for each user in the given list.
    :param preprocess_lst: List of users which need to be updated.
    :return:
    """
    with open(settings.ACCESS_FILE, 'rb') as data_file:
        admin_data = json.load(data_file)
        data_file.close()

    for usr in admin_data["accessDetail"]:
        if str(usr["username"]) in preprocess_lst:
            usr["cropped"] = True

    with open(settings.ACCESS_FILE, 'w+') as data_file:
        data_file.write(json.dumps(admin_data))
        data_file.close()


if __name__ == '__main__':
    # exec_path = os.path.abspath(__file__)
    # dir_name = os.path.dirname(exec_path)
    # os.chdir(dir_name)

    print("Obtaining access list...")
    access_list, preprocess_list, retry_read = get_user_list()
    access_hash, access_changed = admin_file_status()

    if settings.USEPI:
        camera = PiCamera().start()
        lock = AccessControl.LockController()
    else:
        camera = Camera().start()
        lock = None

    faceDetector = FaceDetector()
    authorizer = AccessControl.UserAuthorizer(settings.MAX_DETECTION_SEQUENCE)
    accessLogger = AccessControl.AccessLog(settings.LOG_FILE)
    faceIdentifier = FaceIdentifier()
    faceIdentifier.preprocess(preprocess_list)
    update_access_list(preprocess_list)
    faceIdentifier.train(access_list)

    if settings.DISP_FEED:
        cv2.namedWindow(settings.WINDOW_NAME)
    frame = camera.read()

    while frame is not None:
        faces, eyes = faceDetector.detect(frame)
        det_frame, det_user = faceIdentifier.identify(frame, faces, eyes)
        access_granted, access_user = authorizer.update_user_likelihood(det_user)

        if access_granted and settings.USEPI:
            lock.open()
            accessLogger.log_user(access_user)
            time.sleep(settings.LOCK_TIMEOUT)
            lock.close()

        if access_granted and not settings.USEPI:
            accessLogger.log_user(access_user)
            time.sleep(settings.LOCK_TIMEOUT)

        if settings.DISP_FEED:
            cv2.imshow(settings.WINDOW_NAME, frame)
        frame = camera.read()  # get next frame

        access_hash, access_changed = admin_file_status(access_hash)
        if access_changed or (not access_changed and retry_read):
            print("Retraining models...")
            access_list, preprocess_list, retry_read = get_user_list()
            faceIdentifier.preprocess(preprocess_list)
            update_access_list(preprocess_list)
            faceIdentifier.train(access_list)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(settings.WINDOW_NAME)
    camera.stop()

