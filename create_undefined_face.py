"""
This script was only used to generate a plain black image.
"""

import numpy as np
import cv2
import settings
import os

(im_width, im_height) = (112, 92)

blank_image = np.zeros((im_height, im_width, 3), np.uint8)

for i in range(20):
    path = "{0}/undefined/".format(settings.KEY_FACES)
    if not os.path.isdir(path):
        os.mkdir(path)
    undefined_file = "{0}/undefined/{1}.png".format(settings.KEY_FACES, i + 1)
    cv2.imwrite(undefined_file, blank_image)

