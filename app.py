import os
os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth -P ./weights/")
os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth -P ./weights/")
os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth -P ./weights/")
os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/rrdb_realesrnet_psnr.pth -P ./weights/")

import gradio as gr

'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import glob
import time
import argparse
import numpy as np
from PIL import Image
import __init_paths
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet
from align_faces import warp_and_crop_face, get_reference_facial_points
import torch
torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg', 'mona.jpg')
torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/5/50/Albert_Einstein_%28Nobel%29.png', 'einstein.png')

class FaceEnhancement(object):
    def __init__(self, base_dir='./', size=512, model=None, use_sr=True, sr_model=None, channel_multiplier=2, narrow=1, key=None, device='cuda'):
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.facegan = FaceGAN(base_dir, size, model, channel_multiplier, narrow, key, device=device)
        self.srmodel =  RealESRNet(base_dir, sr_model, device=device)
        self.faceparser = FaceParse(base_dir, device=device)
        self.use_sr = use_sr
        self.size = size
        self.threshold = 0.9
        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")
        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.size, self.size), inner_padding_factor, outer_padding, default_square)
    def mask_postprocess(self, mask, thres=20):
        mask[:thres, :] = 0; mask[-thres:, :] = 0
        mask[:, :thres] = 0; mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        return mask.astype(np.float32)
    def process(self, img):
        if self.use_sr:
            img_sr = self.srmodel.process(img)
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])
        facebs, landms = self.facedetector.detect(img)
        
        orig_faces, enhanced_faces = [], []
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)
        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4]<self.threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])
            facial5points = np.reshape(facial5points, (2, 5))
            of, tfm_inv = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.size, self.size))
            
            # enhance the face
            ef = self.facegan.process(of)
            
            orig_faces.append(of)
            enhanced_faces.append(ef)
            
            #tmp_mask = self.mask
            tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0]/255.)
            tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)
            if min(fh, fw)<100: # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)
            
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)
            mask = tmp_mask - full_mask
            full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
            full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]
        full_mask = full_mask[:, :, np.newaxis]
        if self.use_sr and img_sr is not None:
            img = cv2.convertScaleAbs(img_sr*(1-full_mask) + full_img*full_mask)
        else:
            img = cv2.convertScaleAbs(img*(1-full_mask) + full_img*full_mask)
        return img, orig_faces, enhanced_faces
        
  
    
model = "GPEN-BFR-512"

key = None
size = 512
channel_multiplier = 2
narrow = 1
use_sr = False
use_cuda = False
sr_model = 'rrdb_realesrnet_psnr'

    
    
faceenhancer = FaceEnhancement(size=size, model=model, use_sr=use_sr, sr_model=sr_model, channel_multiplier=channel_multiplier, narrow=narrow, key=key, device='cpu')
    
def inference(file):
    im = cv2.imread(file, cv2.IMREAD_COLOR) 
    img, orig_faces, enhanced_faces = faceenhancer.process(im)
    return enhanced_faces[0][:,:,::-1]
        
title = "GPEN"
description = "Gradio demo for GAN Prior Embedded Network for Blind Face Restoration in the Wild. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2105.06070' target='_blank'>GAN Prior Embedded Network for Blind Face Restoration in the Wild</a> | <a href='https://github.com/yangxy/GPEN' target='_blank'>Github Repo</a></p><center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_GPEN' alt='visitor badge'></center>"


gr.Interface(
    inference, 
    [gr.inputs.Image(type="filepath", label="Input")], 
    gr.outputs.Image(type="numpy", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[
    ['mona.jpg'],
    ['einstein.png']
    ],
    enable_queue=True
    ).launch(debug=True)