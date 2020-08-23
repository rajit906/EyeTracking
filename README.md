# EyeTracking
The model is based on CNNs and essentially predicts where on the screen youre looking at on the screen and moves the cursor

'''bash
python -i <file_name>
'''


Ree.py detects pupil

![pupil_detect](https://user-images.githubusercontent.com/62230387/90968750-ce971a00-e4a4-11ea-8bb2-d18f65d6ca0e.png)

eyetrack.py crops everything out except the left eye

Load.py takes in data from train/ and test/, trains it and loads it into xModels/ and yModels/

![eye_dataset](https://user-images.githubusercontent.com/62230387/90968751-cf2fb080-e4a4-11ea-9123-2268cad0603a.jpg)

TrackandTrain.py takes the model parameters and trains it further to fit labels

GetEyes.py opens webcam and moves cursor around

![reeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee](https://user-images.githubusercontent.com/62230387/90968746-c8a13900-e4a4-11ea-961d-623c7f3a1635.gif)

Note: I messed up some of the number of perceptrons before pushing, that might need some tweaking, otherwise its ready to go!


