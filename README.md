![EarMuff Detected](Sample_detection.jpg?raw=true "earMuff Detected")
![EarMuff NOTDetected](Sample_detection2.jpg?raw=true "earMuff NOTDetected")

# EarMuff_detector

https://user-images.githubusercontent.com/69402254/182837301-fd583c36-9243-4482-a484-59a2a5fbbd10.mp4  


This repo contains an earmuff detector that uses video streams from CCTV cameras in workplaces to detect whether or not a person in the video is wearing earmuffs.   
    <p align="center">
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BunNybuger/EarMuff_detector/blob/master/EarMuff_Detector_using_few_shot_learning_Colab_version.ipynb)
    </p>
We've used YoloV5 models to detect the person in the video and then their head, which is then sent to our PrototypicalNetwork for further analysis. We've trained our network with a small number of images (8 to be exact), and the results are promising. We've also included a sample video for you to check out.
Due to the lack of a dedicated earmuff dataset, we employed a technique called Few Shot Learning (FSL) and used the efficientnet_b2 network architecture pre-trained on the ImageNet dataset as a feature extractor.
We acknowledge that there is still room for improvement, and in future, we will be working on expanding our dataset and testing other methods to improve the accuracy. We'll keep this repository updated with our progress. And a big thanks to @ebennequin for his contribution in Few-Shot Learning.

So, what are you waiting for? Dive in, check out our work and let us know what you think!