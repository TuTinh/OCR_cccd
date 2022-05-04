
from flask import Flask, render_template, request, url_for, redirect
import cv2 
import os 
from read_card import CardReader

app = Flask(__name__)

card_reader = CardReader()  


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['post'])
def upload():
    f = request.files['images']
    img_path = os.path.join(app.root_path, 'static/images/', f.filename)
    f.save(img_path)

    img = cv2.imread(img_path)
    list_recog, img = card_reader(img, img_path, f.filename)

    # list_recog2 = []
    # list_recog2.append(list_recog[1])
    # list_recog2.append(list_recog[2])
    # list_recog2.append(list_recog[6])
    # list_recog2.append(list_recog[7])
    # list_recog2.append(list_recog[8])
    # list_recog2.append(list_recog[10])

    return render_template('index.html', filenames=[f.filename, 'detect_' + f.filename], info=list_recog)


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='images/' +filename), code=301)




if __name__ == '__main__':

    app.run(debug=True)



