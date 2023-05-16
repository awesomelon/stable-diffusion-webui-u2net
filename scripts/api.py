import string
import random
import requests

from fastapi import FastAPI, Body

import gradio as gr
from modules.api.models import *
from modules.api import api

import os
from skimage import io, transform
from skimage.filters import gaussian
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from scripts.data_loader import RescaleT
from scripts.data_loader import ToTensor
from scripts.data_loader import ToTensorLab
from scripts.data_loader import SalObjDataset

from scripts.model import U2NET # full size version 173.6 MB
from scripts.model import U2NETP # small version u2net 4.7 MB

import argparse

from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from dotenv import load_dotenv

SECRET_ID = os.environ.get('SECRET_ID')
SECRET_KEY = os.environ.get('SECRET_KEY')
REGION = os.environ.get('REGION')
BUCKET = os.environ.get('BUCKET')
COS_ENDPOINT = os.environ.get('COS_ENDPOINT')

model_name='u2net_portrait'
model_dir = '/models/u2net_portrait/u2net_portrait.pth'

DEVICE_ID = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0

# Tencent Cloud 인증 및 클라이언트 설정
config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY, Scheme='https')
client = CosS3Client(config)

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def generate_random_string(length=10):
    letters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def u2net_api(_:gr.Blocks, app: FastAPI):
    @app.post("/u2net")
    async def u2net(
        images: List[str] = Body([], description="이미지"),
        sigma: float = Body(20, description="Sigma"),
        alpha: float = Body(0.5, description="Alpha"),
    ):
        results = []
        input_dir = f"/assets/u2net-images/inputs"
        output_dir = f"/assets/u2net-images/outputs"
        prefix = generate_random_string(8)

        print(f"-------------- 1. get image path and name --------------")
        for image in images:
            filename = os.path.basename(image)            
            response = requests.get(image)
            image_name = f"{input_dir}/{prefix}_{filename}"
            with open(image_name, "wb") as f:
                f.write(response.content)
        
        img_name_list = glob.glob(input_dir+f"/{prefix}*")
        print("Number of images: ", len(img_name_list))


        print(f"-------------- 2. data loader --------------")
        salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [], transform=transforms.Compose([RescaleT(512),  ToTensorLab(flag=0)]))
        salobj_dataloader = DataLoader(salobj_dataset, batch_size=1,  shuffle=False, num_workers=1)

        print(f"-------------- 3. model define --------------")
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)

        net.load_state_dict(torch.load(model_dir, map_location=torch.device(f"cuda:{DEVICE_ID}")))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()

        print(f"-------------- 4. inference for each image --------------")
        for i, data in enumerate(salobj_dataloader):
            print("inferencing:",img_name_list[i].split(os.sep)[-1])
            inputs = data['image']
            inputs = inputs.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)

            d1,d2,d3,d4,d5,d6,d7= net(inputs)

            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)

            predict = pred
            predict = predict.squeeze()
            predict_np = predict.cpu().data.numpy()

            image = io.imread(img_name_list[i])
            pd = transform.resize(predict_np,image.shape[0:2],order=2)
            pd = pd/(np.amax(pd)+1e-8)*255            
            pd = pd[:,:,np.newaxis]

            print(image.shape)
            print(pd.shape)

            ## fuse the orignal portrait image and the portraits into one composite image
            ## 1. use gaussian filter to blur the orginal image
            sigma=sigma
            image = gaussian(image, sigma=sigma, preserve_range=True)

             ## 2. fuse these orignal image and the portrait with certain weight: alpha
            alpha = alpha
            im_comp = image*alpha+pd*(1-alpha)
        
            
            file_name = f"{prefix}_{i}.png"
            io.imsave(f"{output_dir}/{file_name}",im_comp)


            print(f"-------------- 5. upload to cos --------------")
            with open(f"{output_dir}/{file_name}", "rb") as f:
                client.put_object(
                    Bucket=BUCKET,
                    Body=f,
                    Key=f"outputs/u2net/{file_name}",
                    StorageClass='STANDARD',
                    EnableMD5=False
                )

                results.append(f"https://{COS_ENDPOINT}/outputs/u2net/{file_name}")
            del d1,d2,d3,d4,d5,d6,d7

        return {"results": results}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(u2net_api)
except:
    pass