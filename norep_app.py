from flask import Flask, render_template, Response, request, url_for
import cv2
import pandas as pd
import mediapipe as mp
import logging
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from helper_funcs import *
from camera import Camera
camera = Camera(fps=60)

logger = logging.getLogger(__name__)

app = Flask(__name__)

camera = Camera(fps=60)
camera.run()



@app.route('/')
def index():
    return render_template('index.html')



def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype="multipart/x-mixed-replace; boundary=frame")



if __name__ == "__main__":
    app.run(debug=False, port=3430)