import os
os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth -P .")
os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth -P .")
os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/ParseNet-latest.pth -P .")
os.system("wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/rrdb_realesrnet_psnr.pth -P .")

import random
import gradio as gr
from PIL import Image
import torch
torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Abraham_Lincoln_O-77_matte_collodion_print.jpg/1024px-Abraham_Lincoln_O-77_matte_collodion_print.jpg', 'lincoln.jpg')
torch.hub.download_url_to_file('https://upload.wikimedia.org/wikipedia/commons/5/50/Albert_Einstein_%28Nobel%29.png', 'einstein.png')
import cv2
import glob
import numpy as np
 

def inference(img):
    os.system("python face_enhancement.py --model GPEN-BFR-512 --size 512 --channel_multiplier 2 --narrow 1 --use_sr --indir examples/imgs --outdir examples/outs-BFR2")
    return "examples/outs-BFR2/"
        
title = "GFP-GAN"
description = "Gradio demo for GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. Please click submit only once"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2101.04061'>Towards Real-World Blind Face Restoration with Generative Facial Prior</a> | <a href='https://github.com/TencentARC/GFPGAN'>Github Repo</a></p><center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_GFPGAN' alt='visitor badge'></center>"
gr.Interface(
    inference, 
    [gr.inputs.Image(type="filepath", label="Input")], 
    gr.outputs.Image(type="pil", label="Output"),
    title=title,
    description=description,
    article=article,
    examples=[
    ['lincoln.jpg'],
    ['einstein.png']
    ],
    enable_queue=True
    ).launch(debug=True)