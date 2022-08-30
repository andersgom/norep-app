# PACKAGES

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import pickle



with open('repcountsquat.p', 'rb') as file:
    model2 = pickle.load(file)


# NOREP APP - MEDIAPIPE STREAM

cap = cv2.VideoCapture(0)

# Counter variables
counter = 0 
grip = None
stage = None

## Setup mediapipe instance

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Model implementation
            poses = results.pose_landmarks.landmark
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
                                
        except:
            pass
        
        
        # Setup status box
        cv2.rectangle(image, (0,0), (225,73), (87,122,59), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (25,15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (30,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 2, cv2.LINE_AA)
         
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
        
        cv2.imshow('NoRep app', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()