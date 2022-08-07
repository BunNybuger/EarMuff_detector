#!/usr/bin/env python
# coding: utf-8

# # Earmuff detector with Few Shot Learning
# This repo is an earmuff detector. It can get video stream from CCTV in workplaces and detect if the person in the video is using an earmuff or not.
# First, a trained YoloV5 model detects the person in the video; then, another YoloV5 model detects the person's head. The detected box of a head is sent to a PrototypicalNetwork, which was not trained to detect earmuff previously. The accuracy is acceptable, as you can see in the sample video. However, it should be noted that I have used some pictures of this video as the support set, and this will increase the accuracy but may not be possible in most practical situations.
# I take the Few Shot Learning(FSL) method because we don't have a dataset. I used efficientnet_b2 network architecture pre-trained on the ImageNet dataset as a feature extractor. In a later work, after a proper dataset is ready for this purpose, another method will be tested in reported here.
# 
# [Here is the FSL repository](https://github.com/sicara/easy-few-shot-learning) from GitHub that I used. Thanks to @ebennequin, there is a good explanation of few-shot learning in the repository as well. you can [access the google colab file here](https://colab.research.google.com/github/sicara/easy-few-shot-learning/blob/master/notebooks/my_first_few_shot_classifier.ipynb)

# In[1]:


import cv2
# import tensorflow as tf 
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import non_max_suppression 
import torch
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression,     apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
import time
import os


# # Few Shot Learning Model: 

# In[2]:


from torchvision import transforms
import torchvision.transforms.functional
convert_tensor = transforms.ToTensor()

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Omniglot
from torchvision.models import efficientnet_b2 #resnet18
from tqdm import tqdm


from easyfsl.data_tools import EasySet, TaskSampler
from easyfsl.utils import plot_images, sliding_average

name_classes = ['Muff', 'NotMuff']
color_list = [(0,255,0),(0,0,255)]


# In[3]:


class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """
        # Extract the features of support and query images
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists
        return scores


convolutional_network = efficientnet_b2(pretrained=True)
convolutional_network.fc = nn.Flatten()
# print(convolutional_network)

model_muff = PrototypicalNetworks(convolutional_network).cuda()


# ### Load Constant Support Dataset:

# In[8]:


C_N_WAY = 2 # Number of classes in a task
C_N_SHOT = 2 # Number of images per class in the support set
C_N_QUERY = 0 # Number of images per class in the query set
C_N_EVALUATION_TASKS = 1

constant_support_set = EasySet(specs_file="./Support Data/ConstantSupportDataset.json", training=False)
test_sampler = TaskSampler(
    constant_support_set, n_way=C_N_WAY, n_shot=C_N_SHOT, n_query=C_N_QUERY, n_tasks=C_N_EVALUATION_TASKS
)

constant_support_data_loader = DataLoader(
    constant_support_set,
    batch_sampler=test_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=test_sampler.episodic_collate_fn,
)

(   constant_support_images,
    constant_support_labels,
    example1_query_images,
    example1_query_labels,
    example1_class_ids,
) = next(iter(constant_support_data_loader))


# ################

# # Load video YoloV5 models

# In[9]:



def resize_with_pad(img, new_w, new_h):

    h,w,_ = img.shape

    if h > w:
        pad_size = h-w
        top, bottom = 0,0
        left, right = pad_size//2, pad_size//2
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value = 0)

    elif w > h:
        pad_size = w-h
        top, bottom = pad_size//2, pad_size//2
        left, right = 0,0
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value = 0)

    img = cv2.resize(img, (new_w,new_h))

    return img


model_PersonDetector = attempt_load('./Person_Detector.pt', map_location='cuda')  # load FP32 model
model_HeadDetector = attempt_load('./Head_detector_fromPersonBox_yolov5.pt', map_location='cuda')


# In[10]:


# ======================= Opens the Video file =======================
video_folder='./Test Video/'
video_name = 'GunShop_Trim'
dataset = LoadImages(video_folder + video_name +'.mp4', img_size=640, stride=64)

expand_headbox_width = 1.2 # multiplied to head box width to include earmuff

cap = cv2.VideoCapture(video_folder + video_name +".mp4")
if (cap.isOpened()== False):
    print("Error opening video file")
video_fps = int(cap.get(5))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# for path, img, im0s, vid_cap, s in dataset:

#     video_fps = int(vid_cap.get(5))
#     frame_width = int(vid_cap.get(3))
#     frame_height = int(vid_cap.get(4))
#     break

frame_save_path = video_folder+video_name
head_save_path = video_folder+video_name +"/head"
Muff_save_path = video_folder + video_name+"/Muff"
NotMuff_save_path = video_folder + video_name+"/NotMuff"

if not os.path.exists(frame_save_path):
    os.mkdir(frame_save_path)

vid_writer = cv2.VideoWriter(frame_save_path +"/" + video_name + '_Out.mp4',cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width,frame_height))


# In[11]:


# ======================= For each frame does prepration and detects =======================

if not os.path.exists(head_save_path):
    os.mkdir(head_save_path)
if not os.path.exists(Muff_save_path):
    os.mkdir(Muff_save_path)
if not os.path.exists(NotMuff_save_path):
    os.mkdir(NotMuff_save_path)

counter = 0
for path, img, im0s, vid_cap, s in dataset:
    counter += 1
    if counter%1==0:
        img = torch.from_numpy(img).to('cuda')
        im0s_tensortorch = img
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  
        if len(img.shape) == 3:
            img = img[None]
        
        pred = model_PersonDetector(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, classes=0)
        person = 0
        for i, det in enumerate(pred):
            if len (pred) >1 :
                print("len(pred) = ",len(pred))
            if i > 0:
                print("first i = ", i)
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                        person += 1
                        px1,py1 , px2,py2 = torch.tensor(xyxy).view(1, 4).view(-1).tolist()

                        img_crop = im0[int(py1):int(py2+1), int(px1):int(px2+1), :]
                        img_in = img_crop[...,::-1] #Convert BGR to RGB(minus1 step size in last dimension)
                        img_in = resize_with_pad(img_in, 640, 640)
                        img_in = np.moveaxis(img_in, -1, 0)

                        img_in = torch.from_numpy(img_in).to('cuda')
                        img_in = img_in.float()
                        img_in = img_in / 255.0

                        if len(img_in.shape) == 3:
                            img_in = img_in[None]

                        pred_head = model_HeadDetector(img_in, augment=False, visualize=False)[0]
                        pred_head = non_max_suppression(pred_head, conf_thres=0.6)

                        for _, det in enumerate(pred_head):
                            p, s, img0, frame = path, '', img_crop.copy(), getattr(dataset, 'frame', 0)
                            if len(det):
                                det[:, :4] = scale_coords(img_in.shape[2:], det[:, :4], img0.shape).round()
                                for *xyxy, conf, cls in reversed(det):
                                        
                                    hx1,hy1,hx2,hy2 = torch.tensor(xyxy).view(1, 4).view(-1).tolist()

                                    x1 = px1 + hx1
                                    y1 = py1 + hy1
                                    x2 = px2 - (img_crop.shape[1]-hx2)
                                    y2 = py2 - (img_crop.shape[0]-hy2)
                                    left_x= x1-((expand_headbox_width-1)*(x2-x1)/2)
                                    right_x=x2+((expand_headbox_width-1)*(x2-x1)/2)
                                    if left_x < 0:
                                        left_x = 0
                                    if right_x > px2:
                                        right_x =px2
                                    # cv2.imwrite(head_save_path +'/'+str(counter)+".jpg",im0s[int(y1):int(y2), int(left_x):int(right_x), :])

                                    img_crop2 = im0[int(y1):int(y2+1), int(left_x):int(right_x+1), :]
                                    img_in2 = img_crop2[...,::-1] #Convert BGR to RGB(minus1 step size in last dimension)
                                    img_in2 = resize_with_pad(img_in2, 224, 224)
                                    img_in2 = np.moveaxis(img_in2, -1, 0)

                                    img_in2 = torch.from_numpy(img_in2).to('cuda')
                                    img_in2 = img_in2.float()
                                    img_in2 = img_in2 / 255.0
                                    
                                    if len(img_in2.shape) == 3:
                                        img_in2 = img_in2[None]
                                                
                                    model_muff.eval()
                                    scores = model_muff(
                                        constant_support_images.cuda(),
                                        constant_support_labels.cuda(),
                                        img_in2,
                                    ).detach()

                                    _, predicted_labels = torch.max(scores.data, 1)

                                    label = name_classes[predicted_labels[0]]
                                    color = color_list[predicted_labels[0]]

                                    cv2.rectangle(im0, (int(left_x), int(y1)), (int(right_x), int(y2)), color, 2)
                                    # cv2.rectangle(im0, (int(x1), int(y1)-10), (int(x1)+20, int(y1)), (0,0,255), -1)
                                    cv2.putText(im0, label, (int(left_x), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
                                    
                                    if label == 'Muff':
                                            cv2.imwrite(Muff_save_path +'/'+str(counter)+"("+str(person)+")"+".jpg",im0s[int(y1):int(y2), int(left_x):int(right_x), :])
                                    else:
                                            cv2.imwrite(NotMuff_save_path +'/'+str(counter)+"("+str(person)+")"+".jpg",im0s[int(y1):int(y2), int(left_x):int(right_x), :])

        cv2.imshow('out', im0)
        # cv2.imwrite(frame_save_path +"/" +str(counter)+".jpg",im0)
        vid_writer.write(im0)
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break

cap.release()
vid_writer.release()
cv2.destroyAllWindows()


# In[ ]:




