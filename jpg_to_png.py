from PIL import Image

import numpy as np
import cv2
import glob
from pathlib import Path
import os

path_img = glob.glob("D:/Image_ID_Card/img_seg_20/*.JPG")



for image in path_img:

      file_name_img = os.path.basename(image)
      temp = file_name_img.split('.')
      file_name = temp[0]
      print(file_name)

      img_jpg = Image.open(image)
      img_jpg.save('./out_png/{}.png'.format(file_name))


