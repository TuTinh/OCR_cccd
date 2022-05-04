from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import cv2
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os

class CardReader:
    def __init__(self):
        self.paddle_ocr = PaddleOCR()
        self.config = Cfg.load_config_from_name('vgg_seq2seq')
        self.config['device'] = 'cpu'
        self.detector = Predictor(self.config)
        # self.width = 600
        # self.height = 440
    
    def __call__(self, img, img_path, filename):
        # if img.shape[0] != self.height or img.shape[1] != self.width:
        #     img = cv2.resize(img, (self.width, self.height))

        img_ori = img.copy()
        detect_text_result = self.paddle_ocr.ocr(img, rec=False)
        
        # delete img in folder
        mydir = './result'
        for f in os.listdir(mydir):
            if not f.endswith(".jpg"):
                continue
            os.remove(os.path.join(mydir, f))

        list_recog = []
        t=0
        for line in detect_text_result:
            t+=1
            # item = {}
            line_arr = np.array(line)
            [x1, y1] = line_arr[0].tolist()
            [x2, y2] = line_arr[2].tolist()
            w = x2 - x1
            h = y2 - y1

            x1 = x1 - 0.03*w if (x1 - 0.03*w) > 0 else 0
            y1 = y1 - 0.1*h if (y1 - 0.1*h) > 0 else 0
            x2 = x2 + 0.03*w if (x2 + 0.03*w) < img.shape[1] else img.shape[1]
            y2 = y2 + 0.1*h if (y2 + 0.1*h) < img.shape[0] else img.shape[0]
            img_crop = img_ori[int(y1):int(y2), int(x1):int(x2)]

            

            cv2.imwrite('./result/result_'+ str(t) +'.jpg', img_crop) 

            if img_crop.shape[0] > 10 and img_crop.shape[1] > 10 and img_crop is not None and img_crop.shape[0] < img_crop.shape[1]:
                img_crop = Image.fromarray(img_crop)
                content = self.detector.predict(img_crop)
                # item['text'] = content
                # item['coordinate'] = [x1, y1, x2, y2]
                list_recog.append(content)
                
        # save img detect
        image = Image.open(img_path).convert('RGB')
        im_show = draw_ocr(image, detect_text_result, txts=None, scores=None, font_path='latin.ttf')
        im_show = Image.fromarray(im_show)

        im_show.save('./static/images/detect_' + filename)
        
        return list_recog, img
