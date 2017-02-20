import cv2
import numpy as np
import settings
import os


class FaceDetector(object):
    """
    This class will be used to determine if there is a face within a given image.
    Here we will use Haar-Like features for the classification. The training for
    this classifier has already been computed and stored in an XML file.
    """

    def __init__(self):
        self._faceCascade = cv2.CascadeClassifier(settings.FACE_CASCADE_FILE)
        self._eyeCascade = cv2.CascadeClassifier(settings.EYE_CASCADE_FILE)

    def extract_face(self, frame):
        """
        Returns a face (as an image) if one can be found.
        :param frame:
        :return:
        """
        (im_width, im_height) = (112, 92)
        im = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gray, (gray.shape[1] / settings.DOWNSAMPLE, gray.shape[0] / settings.DOWNSAMPLE))
        faces = self._faceCascade.detectMultiScale(mini)
        faces = sorted(faces, key=lambda x: x[3])

        if faces:
            face_i = faces[0]
            (x, y, w, h) = [v * settings.DOWNSAMPLE for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            return face_resize
        else:
            return None

    def detect(self, frame):
        """
        Detect faces with the face cascade.
        :param frame:
        :return:
        """
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(grayframe, (frame.shape[1] / settings.DOWNSAMPLE, frame.shape[0] / settings.DOWNSAMPLE))
        faces = self._faceCascade.detectMultiScale(
            mini,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return self.get_face_eye_list(grayframe, faces)

    @staticmethod
    def add_bboxes(frame, faces):
        """
        Adds rectagles around the faces in the current frame.
        :param frame:
        :param faces:
        :return:
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    def get_face_eye_list(self, frame, faces):
        """
        Returns a list of faces which contain two eyes. The idea is that
        a detected face can be better recognized if the eyes are also visible.
        :param frame:
        :param faces:
        :return:
        """
        used_faces = []
        used_eyes = []

        for face in faces:
            (fx, fy, fw, fh) = [v * settings.DOWNSAMPLE for v in face]
            face_crop = frame[fy:fy+fh, fx:fx+fw]
            temp_eyes = self._eyeCascade.detectMultiScale(face_crop)
            for eye in temp_eyes:
                eye[0] = eye[0] + fx
                eye[1] = eye[1] + fy

            # If more than two eyes are found, find the pair of eyes closest in size, ignore other eyes
            if len(temp_eyes) > 2:
                diffs = []
                index_1 = []
                index_2 = []
                for e in range(0, len(temp_eyes) - 1):
                    for x in range(e + 1, len(temp_eyes)):
                        diff = abs(temp_eyes[e][2] - temp_eyes[x][2])
                        index_1.append(e)
                        index_2.append(x)
                        diffs.append(diff)
                i = diffs.index(min(diffs))
                used_eyes.append(temp_eyes[index_1[i]])
                used_eyes.append(temp_eyes[index_2[i]])
            elif len(temp_eyes) == 2:
                [used_eyes.append(e) for e in temp_eyes]
                used_faces.append(face)

        return used_faces, used_eyes


class FaceIdentifier(object):
    """
    This class will associate a face as a certain person.
    """

    def __init__(self):
        (self._im_width, self._im_height) = (112, 92)
        self._model = cv2.face.createLBPHFaceRecognizer(threshold=settings.FACE_THRESHHOLD)
        # self._model = cv2.face.createEigenFaceRecognizer()
        self._labels = None
        self._names = None

    def preprocess(self, img_list):
        """
        The users which have not been in the system before need to be preprocessed
        because the images that are created by the web server will most likely contain
        background. This method will crop the face in the image and replace the original file.
        :param img_list:
        :return:
        """
        face_detector = FaceDetector()
        fn_dir = settings.KEY_FACES

        for usr in img_list:
            usr_dir = os.path.join(fn_dir, usr)
            for usr_img in os.listdir(usr_dir):
                if usr_img.endswith(".png"):
                    rmv_name = usr_img
                    new_name = os.path.splitext(usr_img)[0] + "_cropped" + os.path.splitext(usr_img)[1]
                    img_data = cv2.imread("{}/{}".format(usr_dir, usr_img))
                    cropped_img = face_detector.extract_face(img_data)
                    if cropped_img is not None:
                        cv2.imwrite("{}/{}".format(usr_dir, new_name), cropped_img)
                    os.remove("{}/{}".format(usr_dir, rmv_name))
        return

    def train(self, access_list):
        """
        Trains the face recognizer based on which users have been authorized in the 'access_list'.
        :param access_list:
        :return:
        """
        fn_dir = settings.KEY_FACES

        # Create fisherRecognizer
        print('Training...')
        # Create a list of images and a list of corresponding names
        (images, self._labels, self._names, id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(fn_dir):
            for subdir in dirs:
                if subdir in access_list:
                    self._names[id] = subdir
                    subjectpath = os.path.join(fn_dir, subdir)
                    for filename in os.listdir(subjectpath):
                        if filename.endswith(".png"):  # only use image files
                            path = subjectpath + '/' + filename
                            label = id
                            images.append(cv2.imread(path, 0))
                            self._labels.append(int(label))
                    id += 1

        # Create a Numpy array from the two lists above
        (images, self._labels) = [np.array(lis) for lis in [images, self._labels]]

        self._model.train(images, self._labels)
        print('Training Finished')
        return

    def identify(self, frame, faces, eyes):
        """
        Attempts to identify the faces within a frame. The IDs and scores are stored
        for predicting the most likely user in order to successfully authorize them.
        The faces & eyes are annotated in the frame for visual inspection.
        :param frame:
        :param faces:
        :param eyes:
        :return:
        """

        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        radius = 20
        detections = {}

        for i in range(len(faces)):
            face_i = faces[i]
            (x, y, w, h) = [v * settings.DOWNSAMPLE for v in face_i]
            face = grayframe[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (self._im_width, self._im_height))

            # Try to recognize the face
            prediction = self._model.predict(face_resize)

            cv2.rectangle(frame, (x + radius, y + radius), (x + w, y + h), (0, 255, 0), 3)

            eye1 = eyes[i]
            (x1, y1, w1, h1) = [v for v in eye1]
            cv2.circle(frame, (x1 + radius, y1 + radius), radius, (155, 55, 200), 2)

            eye2 = eyes[i+1]
            (x2, y2, w2, h2) = [v for v in eye2]
            cv2.circle(frame, (x2 + radius, y2 + radius), radius, (155, 55, 200), 2)

            if prediction[0] < 0: # settings.FACE_THRESHHOLD:
                print("UNKOWN --> ID: {} Confidence: {}".format(settings.UNKNOWN_USER, prediction[1]))
                detections[settings.UNKNOWN_USER] = prediction[1]  # save dist. of all faces
                cv2.putText(frame, settings.UNKNOWN_USER,
                            (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            else:
                print("ID: {} Confidence: {}".format(self._names[prediction[0]], prediction[1]))
                detections[self._names[prediction[0]]] = prediction[1]  # save dist. of all faces
                cv2.putText(frame,
                            '%s - %.0f' % (self._names[prediction[0]], prediction[1]),
                            (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        # find the most likely users based on the distances of the detected users
        lik_user = []
        if len(detections.values()) > 0:
            min_dist_val = min(detections.values())
            lik_user = [usr for usr, dist in detections.iteritems() if dist == min_dist_val]

        return frame, lik_user

