
import numpy as np
import cv2
import glob
from pathlib import Path
import os
import json


# points = np.array([[[468.38235294117646,483.8235294117647],[463.9705882352941,502.9411764705882],[466.66666666666663,832.2222222222222],
#         [
#           472.22222222222223,
#           1390.0
#         ],
#         [
#           476.1111111111111,
#           1402.5
#         ],
#         [
#           483.6111111111111,
#           1413.3333333333333
#         ],
#         [
#           490.55555555555554,
#           1420.8333333333333
#         ],
#         [
#           502.22222222222223,
#           1428.3333333333333
#         ],
#         [
#           518.8888888888889,
#           1433.611111111111
#         ],
#         [
#           566.1111111111111,
#           1434.4444444444443
#         ],
#         [
#           1127.8350515463917,
#           1427.4914089347078
#         ],
#         [
#           1964.2335766423357,
#           1419.7080291970801
#         ],
#         [
#           1983.0423940149626,
#           1410.4738154613467
#         ],
#         [
#           1995.2618453865339,
#           1397.2568578553617
#         ],
#         [
#           2004.7381546134663,
#           1383.5411471321697
#         ],
#         [
#           2009.9750623441398,
#           1372.5685785536161
#         ],
#         [
#           1997.8835978835978,
#           515.8730158730158
#         ],
#         [
#           1991.005291005291,
#           492.59259259259255
#         ],
#         [
#           1979.8941798941798,
#           479.36507936507934
#         ],
#         [
#           1965.079365079365,
#           468.2539682539682
#         ],
#         [
#           1950.7936507936506,
#           464.021164021164
#         ],
#         [
#           1938.6243386243384,
#           462.96296296296293
#         ],
#         [
#           1411.6402116402116,
#           461.90476190476187
#         ],
#         [
#           510.57142857142856,
#           456.57142857142856
#         ],
#         [
#           495.42857142857144,
#           461.14285714285717
#         ],
#         [
#           485.14285714285717,
#           468.2857142857143
#         ],
#         [
#           475.7142857142857,
#           474.85714285714283
#         ]
#       ]])

if __name__ == '__main__':
    
    path_img = glob.glob("D:/Image_ID_Card/img_seg_20/*.JPG")
    path_save = "./Output_img_Mask/"
    Path("Output_img_Mask").mkdir(parents=True, exist_ok=True)

    ## read image and json in folder
    for image in path_img:

      file_name_img = os.path.basename(image)
      temp = file_name_img.split('.')

      file_name = temp[0]
      print(file_name)

      # get point in file json
      file_json = open('D:/Image_ID_Card/img_seg_20/{}.json'.format(file_name), "r")
      data_json = json.loads(file_json.read())
      data_shapes = data_json['shapes']
      data_point = data_shapes[0].get('points')
      print(data_point)

      points = np.array(data_point)

      # image process
      img = cv2.imread(image)
      mask = np.zeros(img.shape[0:2], dtype=np.uint8)
      points = np.array(points).reshape((-1,1,2)).astype(np.int32)
    
      cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
      cv2.imwrite("Mask.png", mask)        

    
      cv2.imwrite(os.path.join(path_save , '{}.png'.format(file_name)), mask)

      file_json.close()
    
    
      # t+=1
      # print('===================== {} ----------------'.format(t))    
              


