# **NoRep App**

### Fitness Web App focused on weightlifting that tracks your workout using Computer Vision and gives you feedback on your posture.

<br>

### **The idea**

After training via Zoom for months, I noticed that fitness coaches had limited visibility through the webcam (especially during group classes). This app is meant to be a tool for fitness professionals that, during online classes, will allow them to spot bad practices in real time and improve the performance of their coachees.

This project is powered by the amazing technology of [Mediapipe](https://mediapipe.dev/).

Check the presentation [here](https://www.canva.com/design/DAFK_LJqxA4/uFmDz_rNq8PuCC3Y5FaNMw/view?utm_content=DAFK_LJqxA4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

<br>

### **Features**

- **Pose recognition**:
    - ***Deadlift***: Repetition counter and posture assistant.
    - ***Air squat***: Repetition counter.

<br>

- **Time-out!**:
    - The recognition model only initializes when there are people in front of the camera, to prevent false repetitions to appear.

<br>

- **Metrics report**:
    - Once your workout is done, you'll find a report with your repetitions in each movement and the accuracy of your posture.

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

The data was taken from videos of my own workouts, assisted by a certified Crossfit coach, and online videos to train the model with more diverse body shapes and environments. To identify each position, I made manual labelling. The output dataset is composed of the coordinates of 33 points, each one of them a different body part.

<br>

### **The model**

Because the data is composed of coordinate points, I used a `K-Nearest Neighbors (KNN)` algorithm for the model, tuning the hyperparameters to give more priority to the points that are closest to each other. To improve the model, I did signal boosting on the model by reaching out to fitness professionals.

<br>

### **Future steps**

- Increment the number of movements available.
- Deployment in Azure.
- Database and login implementation.
- Potential deployment in video conferencing platforms (i.e. Zoom, Skype)

<br>

### **Preview**

![Preview](https://raw.githubusercontent.com/andersgom/norep-app/main/static/videos/NoRep-App-Demo-gif.gif?token=GHSAT0AAAAAABYIZV6M77YXEOAVUY2UTW36YYQSXCQ)

<br>

### **Acknowledgments**

This project was developed during my time at the Ironhack Data Analytics Bootcamp in Lisbon during the summer of '22. I would like to thank Ironhack Lisbon for the opportunity, José and Gladys for being the best teacher and teacher assistant I could ask for, for their patience and huge commitment, and my Data class for being a group of amazing people and exceptional professionals, and for the bitoques we enjoyed each Friday.

Last but not least, thanks to The Bakery Crossfit for the support and for giving me the inspiration for this project, especially to Edu and Sara for sharing your knowledge and advice with me. In your own way, each one of you has helped me to grow as a person in many different ways, challenged me to improve, and inspired me to become a better person. ***THANK YOU***. 💙
