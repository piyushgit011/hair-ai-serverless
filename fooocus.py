import numpy as np
import requests
import json
from PIL import Image as im
import base64
import uuid
import time
# UQRNCZOKKJXHKPDB5WGWR1RDJSM6PKS7E00ZCSOP

class RunpodAPI:
    def __init__(self, api_key):
        # self.base_url = "https://api.runpod.ai/v2/j6ctvdmtlle2lt/"
        self.base_url = "https://api.runpod.ai/v2/ufgl9xxzkoawwk/"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def run_job(self, input_data):
        url = f"{self.base_url}run"
        data = {"input": input_data}
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()

    def get_job_status(self, job_id):
        url = f"{self.base_url}status/{job_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()
    

def fooocus_endpoint(image_url, mask_url, hairstyle, colour=None):
   api_key = "UQRNCZOKKJXHKPDB5WGWR1RDJSM6PKS7E00ZCSOP"
   runpod_api = RunpodAPI(api_key)
   hairstyle += " hairstyle"  
   hairstyle += f", {colour} colour hair"
   input = {
        "api_name": "inpaint-outpaint",
        "prompt": hairstyle,
        "input_image": image_url,
        "input_mask": mask_url,
        # "image_prompts": [
        #     {
        #         "cn_img": f"{img1}",
        #         "cn_stop":1,
        #         "cn_weight":1.2,
        #         "cn_type":"FaceSwap"
        #     }
        # ],
        "require_base64": True,
        "advanced_params": {"mixing_image_prompt_and_vary_upscale":True,
                            "inpaint_respective_field": 1,
                            "inpaint_strength": 1,
                            "inpaint_erode_or_dilate": 8
        }
   } 
   job_response = runpod_api.run_job(input)
   if "id" in job_response:
        job_id = job_response["id"]
        while True:
            status_response = runpod_api.get_job_status(job_id) 
            if status_response['status']=='COMPLETED':
                break
            time.sleep(2)
   image_data = base64.b64decode(status_response['output'][0]['base64'])
   return image_data
   


def fooocus_endpointt(image_url, mask_url, hairstyle, colour=None):
    if colour:
     hairstyle += " hairstyle"  
     hairstyle += f", {colour} colour hair"
     params = {
        "prompt": hairstyle,
        "style_selections": ["Fooocus V2,Fooocus Enhance,Fooocus Sharp, Fooocus Negative, Fooocus Masterpiece"],
        "input_image": image_url,
        "input_mask": mask_url,
        "require_base64": False,
        "async_process": False,
        # "sharpness": 15,
        "advanced_params": {
            "inpaint_respective_field": 1,
            "inpaint_strength": 1,
            "inpaint_erode_or_dilate": 8
        }
        # "image_prompts": [
        #     {
        #         "cn_img": image_url,
        #         "cn_stop": 0.2,
        #         "cn_weight": 0.2,
        #         "cn_type": "PyraCanny"
        #     }
        # ]
     }
     response = requests.post(url=f"https://84cu79bumd9cos-7000.proxy.runpod.net/v2/generation/image-inpaint-outpaint",
                             data=json.dumps(params))   
    else:
     params = {
        "prompt": hairstyle,
        "style_selections": ["Fooocus V2,Fooocus Enhance,Fooocus Sharp, Fooocus Negative, Fooocus Masterpiece"],
        "input_image": image_url,
        "input_mask": mask_url,
        "advanced_params": {
            "mixing_image_prompt_and_inpaint": True,
            "inpaint_respective_field": 1,
            "inpaint_strength": 1,
            "inpaint_erode_or_dilate": 8
            # "inpaint_erode_or_dilate": 30
        },
        "require_base64": False,
        "async_process": False,
        # "sharpness": 15,
        "image_prompts": [
            {
                "cn_img": image_url,
                "cn_stop": 0.5,
                "cn_weight": 0.4,
                "cn_type": "ImagePrompt"
            }
        ]
     }
     response = requests.post(url=f"https://84cu79bumd9cos-7000.proxy.runpod.net/v2/generation/image-prompt",
                             data=json.dumps(params))   
    return response.json()[0]['url']




