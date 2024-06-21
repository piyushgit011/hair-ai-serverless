import supervision as sv
import torch
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import warnings
from typing import List
import cv2
import numpy as np
from PIL import Image
from fooocus import fooocus_endpoint #,volume_endpoint
from PIL import Image
import requests
from io import BytesIO
import os
import numpy as np
import torch
import cv2
from firebase import convert_to_url
import time
import os
import base64
import io
import runpod 


DEVICE = torch.device('cuda')
GROUNDING_DINO_CHECKPOINT_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

sam_checkpoint = "sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth"
model_type = "vit_l"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)
CLASSES = ['bald']
BOX_TRESHOLD = 0.40
TEXT_TRESHOLD = 0.25


def read_image_from_base64(base64_str: str) -> Image.Image:
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

def preprocess_image_for_opencv(image_pil: Image.Image) -> np.ndarray:
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def process_input(input):
        """
        Execute the application code
        """
        base64 = input['base64']
        hairstyle = input['hairstyle']
        colour = input['colour']
        CLASSES = ['hair']
        BOX_TRESHOLD = 0.40
        TEXT_TRESHOLD = 0.25
        image_base = read_image_from_base64(base64)
        image = preprocess_image_for_opencv(image_base)
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        mask = detections.mask[0]
        mask = (mask.astype(np.uint8) * 255)
        CLASSES = ['person']
        BOX_TRESHOLD = 0.40
        TEXT_TRESHOLD = 0.25
        # image_base = read_image_from_base64(data.image_base64)
        # image = preprocess_image_for_opencv(image)
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        mask1 = detections.mask[0]
        mask1 = (mask1.astype(np.uint8) * 255)
        mask1 = cv2.bitwise_not(mask1)
        mask = cv2.bitwise_or(mask,mask1)
        _, buffer = cv2.imencode('.png', mask)
        mask_data = BytesIO(buffer)
        cloud_file_path = f"masks/mask_{int(time.time())}.png"
        mask_url = convert_to_url(mask_data, cloud_file_path)
        # prompt = prompt + ", hair behind back,"
        base64 = fooocus_endpoint(base64,mask_url,hairstyle,colour)
        # return link
        return {
           "result": base64
        }   

def handler(event):
    """
    This is the handler function that will be called by RunPod serverless.
    """
    return process_input(event['input'])

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
