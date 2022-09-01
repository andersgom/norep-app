# **NoRep App**

### Fitness Web App focused on weightlifting that tracks your body using Computer Vision and gives you feedback on your posture.

<br>

### **The idea**

After training via Zoom for months, I noticed that fitness coaches had a limited visilibity through the webcam (especially during group classes).

This app is meant to be a tool for fitness professionals that, during online classes, that will allow them to spot bad practices in real time and improve the performance of their coachees.

<br>

### **Features**

- **Pose recognition**:
    - ***Deadlift***: Repetition counter and posture assistant.
    - ***Air squat***: Repetition counter.

<br>

- **No body, No Reps**:
    - The recognition model only initializes when there are people in front of the camera, to prevent false repetitions to appear.

<br>

- **Metrics report**:
    - Once your're finished your workout, you'll find a report with your repetitions in each movement and the accuracy of your posture.

<br>

### **Tech Stack**

- Python
- Machine Learning
- Flask
- OpenCV
- Mediapipe
- Main libraries: `cv2`, `Flask`, `mediapipe`, `numpy`.

<br>

### **Data gathering**

The data was taken from videos of my own workouts, assisted by a certified Crossfit coach, and online videos to train the model with more diverse body shapes and environments. To identify each position I made manual labelling.
The output dataset is composed by coordinates of 33 points, being each one of them a different body part.

<br>

### **The model**

Because the data is composed by coordinate points, I used a `K-Nearest Neighbors (KNN)` algorithm  for the model, tuning the hyperparamenters to give more priority to the points that are closer to each other. To improve the model, I did signal boosting on the model by reaching out fitness professionals.

<br>

### **Future steps**

- Increment the number of movements available.
- Deployment in Azure.
- Database and login implementation.
- Potential deployment in video conferencing platforms (i.e. Zoom, Skype)

<br>

### **Preview**

![Preview](static\videos\NoRep-App-Demo-gif.gif)