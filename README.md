# EmoDetect

The main file is PyEmoDetect.py
It contains all functions needed for affect detection.
The normal user will only need two functions:
emo_detect(), which is applied to a list of texts and returns either a pandas dataframe or a numpy array containing raw affect values
compute_metrics(), which is applied to the output of emo_detect() and computes percentage and average values based on the raw values.
