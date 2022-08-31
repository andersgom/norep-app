import cv2
from flask import Flask, render_template, Response
import mediapipe as mp
from helper_funcs import *
import numpy as np
import pandas as pd
import pickle
import warnings


warnings.filterwarnings(action="ignore", category=UserWarning)

app = Flask(__name__)

stance_acc = []
grip_acc = []
counter_df = 0
counter_as = 0

# Models
with open('repcounter.p', 'rb') as file:
    model = pickle.load(file)
with open('repcountsquat.p', 'rb') as file:
    model2 = pickle.load(file)


def gen():
    global stance_acc
    global grip_acc
    global counter_df
    counter = 0 
    grip = None
    stance = None
    stage = None
    grip_acc = []
    stance_acc = []
    # creating our model to draw landmarks
    mp_drawing = mp.solutions.drawing_utils
    # creating our model to detected our pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        # Image processing

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.pose_landmarks:

            # Grip variables
            l_shoulder = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            r_shoulder = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            l_elbow = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            r_elbow = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            # Stance variables
            l_hip = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            r_hip = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            l_ankle = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            r_ankle = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # Calculate angles
            l_grip = calculate_angle(r_shoulder, l_shoulder, l_elbow)
            r_grip = calculate_angle(l_shoulder, r_shoulder, r_elbow)
            l_stance = calculate_angle(r_hip, l_hip, l_ankle)
            r_stance = calculate_angle(l_hip, r_hip, r_ankle)

            # Grip logic
            if (r_grip>90)|(l_grip>90)&(r_grip<120)|(l_grip<120):
                grip = 'Grip: Good!'
                posturebox = cv2.rectangle(image, (0,150), (225,73), (160,170,80), -1)
            if (r_grip>120)|(l_grip>120):
                grip = 'Grip: Too wide'
                posturebox = cv2.rectangle(image, (0,150), (225,73), (80,160,170), -1)
            if (r_grip<90)|(l_grip<90):
                grip = 'Grip: Too narrow'
                posturebox = cv2.rectangle(image, (0,150), (225,73), (80,160,170), -1)
            
            # Grip accuracy logic
            if grip == 'Grip: Good!':
                grip_acc.append(1)
            if (grip == 'Grip: Too wide')|(grip == 'Grip: Too narrow'):
                grip_acc.append(0)

            # Stance logic
            if (r_stance>88)|(l_stance>88)&(r_stance<98)|(l_stance<98):
                stance = 'Stance: Good!'
                posturebox = cv2.rectangle(image, (0,150), (225,73), (160,170,80), -1)
            if (r_stance>98)|(l_stance>98):
                stance = 'Stance: Too wide'
                posturebox = cv2.rectangle(image, (0,150), (225,73), (80,160,170), -1)
            if (r_stance<88)|(l_stance<88):
                stance = 'Stance: Too narrow'
                posturebox = cv2.rectangle(image, (0,150), (225,73), (80,160,170), -1)
            
            # Stance accuracy logic

            if stance == 'Stance: Good!':
                stance_acc.append(1)
            if (stance == 'Stance: Too wide')|(stance == 'Stance: Too narrow'):
                stance_acc.append(0)

            # Model implementation
            poses = result.pose_landmarks.landmark
            pose_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in poses]).flatten()
            frame_height, frame_width = frame.shape[:2]
            pose_row = pose_row * np.array([frame_width, frame_height, frame_width])[:,None]
            X = pd.DataFrame([pose_row[0]])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]

            # Rep counter logic
            if body_language_class == 0:
                stage = 'Down'
            if (body_language_class == 1)&(stage=='Down'):
                stage = 'Up'
                counter +=1

            counter_df = counter

            # Stream Display

            cv2.rectangle(image, (0,0), (225,73), (25,80,25), -1)
            postureboxlogic = posturebox
            postureboxlogic
            
            # Rep data
            cv2.putText(image, 'REPS', (25,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (30,55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
            
            # Posture data
            cv2.putText(image, 'POSTURE', (70,90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, grip, (15,115), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(image, stance, (15,140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (145,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (130,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, f'CONF:{str(round(body_language_prob[np.argmax(body_language_prob)],2))}', (130,68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,40,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(140,180,140), thickness=2, circle_radius=2)  
                                    )

        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break
        
    return stance_acc, grip_acc, counter_df


def gen2():
    global counter_as
    counter = 0 
    stage = None
    # creating our model to draw landmarks
    mp_drawing = mp.solutions.drawing_utils
    # creating our model to detected our pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        # Image processing

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if result.pose_landmarks:

            # Model implementation
            poses = result.pose_landmarks.landmark
            pose_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in poses]).flatten()
            frame_height, frame_width = frame.shape[:2]
            pose_row = pose_row * np.array([frame_width, frame_height, frame_width])[:,None]
            X = pd.DataFrame([pose_row[0]])
            body_language_class = model2.predict(X)[0]
            body_language_prob = model2.predict_proba(X)[0]

            # Rep counter logic
            if body_language_class == 0:
                stage = 'Down'
            if (body_language_class == 1)&(stage=='Down'):
                stage = 'Up'
                counter +=1

            counter_as = counter
            
            # Stream Display

            cv2.rectangle(image, (0,0), (225,73), (25,80,25), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (25,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (30,55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (145,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (130,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, f'CONF:{str(round(body_language_prob[np.argmax(body_language_prob)],2))}', (130,68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,40,0), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(140,180,140), thickness=2, circle_radius=2)  
                                    )

        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break
    
    return counter_as


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/deadlift_feed')
def deadlift_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sq')
def indexsq():
    """Video streaming squat page."""
    return render_template('index_sq.html')

@app.route('/squat_feed')
def squat_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics():
    return render_template('index_metrics.html', countdf = counter_df, countas = counter_as, grip = round(np.mean(grip_acc),1)*100, stance = round(np.mean(stance_acc),1)*100)

if __name__=="__main__":
    app.run(debug=True)