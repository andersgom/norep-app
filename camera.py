import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import pickle
import logging
import threading
from helper_funcs import calculate_angle


with open('repcounter.p', 'rb') as file:
    model = pickle.load(file)
logger = logging.getLogger(__name__)
thread = None

class Camera:
    def __init__(self, fps=20, video_source=0):
        self.fps = fps
        self.video_source = video_source
        self.camera = cv2.VideoCapture(self.video_source)
        # We want a max of 5s history to be stored, thats 5s*fps
        self.max_frames = 5 * self.fps
        self.frames = []
        self.isrunning = False

    def run(self):
        global thread
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.isrunning = True
        thread.start()

    def _capture_loop(self):
        counter = 0 
        grip = None
        stance = None
        stage = None

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.camera.isOpened():
                ret, frame = self.camera.read()

                if(ret == True):
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    # Make detection
                    results = pose.process(image)
                    # Recolor back to BGR
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    print("Inactive")
                    print(ret)
                    

                #try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                # Grip
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                # Stance
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                
                # Calculate angles
                l_grip = calculate_angle(r_shoulder, l_shoulder, l_elbow)
                r_grip = calculate_angle(l_shoulder, r_shoulder, r_elbow)
                l_stance = calculate_angle(r_hip, l_hip, l_ankle)
                r_stance = calculate_angle(l_hip, r_hip, r_ankle)


                # Grip logic
                if (r_grip>90)|(l_grip>90)&(r_grip<120)|(l_grip<120):
                    grip = 'Grip: Good!'
                    posturebox = cv2.rectangle(image, (0,150), (225,73), (200,200,16), -1)
                if (r_grip>120)|(l_grip>120):
                    grip = 'Grip: Too wide'
                    posturebox = cv2.rectangle(image, (0,150), (225,73), (0,145,218), -1)
                if (r_grip<90)|(l_grip<90):
                    grip = 'Grip: Too narrow'
                    posturebox = cv2.rectangle(image, (0,150), (225,73), (0,145,218), -1)

                # Stance logic
                if (r_stance>88)|(l_stance>88)&(r_stance<98)|(l_stance<98):
                    stance = 'Stance: Good!'
                    posturebox = cv2.rectangle(image, (0,150), (225,73), (200,200,16), -1)
                if (r_stance>98)|(l_stance>98):
                    stance = 'Stance: Too wide'
                    posturebox = cv2.rectangle(image, (0,150), (225,73), (0,145,218), -1)
                if (r_stance<88)|(l_stance<88):
                    stance = 'Stance: Too narrow'
                    posturebox = cv2.rectangle(image, (0,150), (225,73), (0,145,218), -1)

                # Model implementation
                poses = results.pose_landmarks.landmark
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
                                        
                #except:
                #    pass

                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (87,122,59), -1)
                postureboxlogic = posturebox
                postureboxlogic
                
                # Rep data
                cv2.putText(image, 'REPS', (25,15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (30,60), 
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
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0,30,0), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(187,225,160), thickness=2, circle_radius=2)  
                                        )


    def stop(self):
        self.isrunning = False

    
    def get_frame(self):
        ret, fram = self.camera.read()

        ret, jpeg = cv2.imencode('.jpg', fram)
        return jpeg.tobytes()





