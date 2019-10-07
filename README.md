# EmoDetect

The main file is PyEmoDetect.py

It contains all functions needed for affect detection.
The normal user will only need two functions:
emo_detect(), which is applied to a list of texts and returns either a pandas dataframe or a numpy array containing raw affect values, and compute_metrics(), which is applied to the output of emo_detect() and computes percentage and average values based on the raw values.
A detailed description of these functions can be found within PyEmoDetect.py

The different files ending in _dict are affect dictionaries in pickle format. 
They are loaded by emo_detect(), and match words to affect ratings or categories.
They need to be present in the same folder as PyEmoDetect.py 
