![EarMuff Detected](Sample detection.jpg?raw=true "earMuff Detected")
![EarMuff Detected](Sample detection.jpg?raw=true "earMuff NOTDetected")

# EarMuff_detector 
https://github.com/BunNybuger/EarMuff_detector/blob/master/Test%20Video/GunShop_Trim_OUTPUT.mp4
sasd
https://user-images.githubusercontent.com/69402254/182837301-fd583c36-9243-4482-a484-59a2a5fbbd10.mp4

This repo is an earmuff detector. It can get video stream from CCTV in workplaces and detect if the person in the video is using an earmuff or not.
First, a trained YoloV5 model detects the person in the video; then, another YoloV5 model detects the person's head. The detected box of a head is sent to a PrototypicalNetwork, which was not trained to detect earmuff previously and it only has seen 8 pictures of a person with and without muff. With only 8 pictures the accuracy is really good, as you can see in the sample video, GunShop_Trim_Out (1).mp4. However, it should be noted that those 8 pictuers are taken from the same test video, and this have increased the accuracy.
I took the Few Shot Learning(FSL) method because we don't have a dataset for this project, yet! I used efficientnet_b2 network architecture pre-trained on the ImageNet dataset as a feature extractor. In a later work, after a proper dataset is ready for this purpose, another method will be tested in reported here.

[Here is the FSL repository](https://github.com/sicara/easy-few-shot-learning) from GitHub that I used. Thanks to @ebennequin, there is a good explanation of few-shot learning in the repository as well. you can [access the google colab file here](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb) 
 
 
