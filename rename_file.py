from PIL import Image

import numpy as np
import cv2
import glob
from pathlib import Path
import os

path_img = glob.glob("C:/Users/tutin/OneDrive/Máy tính/Data_IDCard/*.jpg")


t = 1
for image in path_img:
    # print(image)

      os.rename(image, "C:/Users/tutin/OneDrive/Máy tính/Data_IDCard/image_{}.jpg".format(str(t)))
      t+=1