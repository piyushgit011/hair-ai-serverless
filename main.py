# import supervision as sv
# import torch
# from groundingdino.util.inference import Model
# from segment_anything import sam_model_registry, SamPredictor
# import warnings
# from typing import List
# import cv2
# import numpy as np
# from PIL import Image
# from fooocus import fooocus_endpoint #,volume_endpoint
# from PIL import Image
# import requests
# from io import BytesIO
# import os
# import numpy as np
# import torch
# # import matplotlib.pyplot as plt
# import cv2
# from firebase import convert_to_url
# import time
# # from urllib.parse import parse_qs
# import json
# from fastapi import FastAPI, Request
# import tempfile
# import os
# import base64
# from fastapi import HTTPException
# from fastapi import FastAPI, Response
# import requests
# import io
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI
# from typing import Optional
# from fastapi import FastAPI, UploadFile, HTTPException
# from fastapi import FastAPI, Request, File, UploadFile
# from fastapi.responses import JSONResponse
# import requests
# import re
# from pydantic import BaseModel
# from typing import Optional



# DEVICE = torch.device('cuda')
# GROUNDING_DINO_CHECKPOINT_PATH = "/workspace/GroundingDINO/weights/groundingdino_swint_ogc.pth"
# GROUNDING_DINO_CONFIG_PATH = "/workspace/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# sam_checkpoint = "/workspace/sam-hq/pretrained_checkpoint/sam_hq_vit_l.pth"
# model_type = "vit_l"
# device = "cuda"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# sam_predictor = SamPredictor(sam)
# CLASSES = ['bald']
# BOX_TRESHOLD = 0.40
# TEXT_TRESHOLD = 0.25

# app = FastAPI()

# #app.add_middleware(
# #    CORSMiddleware,
# #    allow_origins=origins,
# #    allow_credentials=True,
# #    allow_methods=["GET", "POST", "OPTIONS"],
# #    allow_headers=["*"],
# #)
# headers = {
#     "accept": "application/json"
# }


# class ImageData(BaseModel):
#     image_base64: str
#     prompt: str
#     colour: Optional[str] = None

# def read_image_from_base64(base64_str: str) -> Image.Image:
#     image_data = base64.b64decode(base64_str)
#     image = Image.open(io.BytesIO(image_data))
#     return image

# def preprocess_image_for_opencv(image_pil: Image.Image) -> np.ndarray:
#     if image_pil.mode != 'RGB':
#         image_pil = image_pil.convert('RGB')
#     image_np = np.array(image_pil)
#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#     return image_bgr

# def enhance_class_name(class_names: List[str]) -> List[str]:
#     return [
#         f"all {class_name}s"
#         for class_name
#         in class_names
#     ]

# @app.post("/wig")
# async def bald(data: ImageData):
#     image_base = read_image_from_base64(data.image_base64)
#     # image = cv2.imread(image_path)
#     image = preprocess_image_for_opencv(image_base)
#     detections = grounding_dino_model.predict_with_classes(
#         image=image,
#         classes=enhance_class_name(class_names=CLASSES),
#         box_threshold=BOX_TRESHOLD,
#         text_threshold=TEXT_TRESHOLD
#     )
#     for x1, y1, x2, y2 in detections.xyxy:
#      y2 = y2 + 80
#      x1 = x1 - 80
#      x2 = x2 + 80
#      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#      width = x2 - x1
#      height = y2 - y1

#      if width > height:
#         diff = (width - height) // 2
#         # y1 = max(y1 - diff, 0)
#         y2 = min(y2 + diff, image.shape[0])
#      elif height > width:
#         diff = (height - width) // 2
#         x1 = max(x1 - diff, 0)
#         x2 = min(x2 + diff, image.shape[1])
#      width = x2 - x1
#      height = y2 - y1
#      if width > height:
#         y2 += 1
#      elif height > width:
#         x2 += 1
#      cutout = image[y1:y2, x1:x2]
#      resized_cutout = cv2.resize(cutout, (1024, 1024), interpolation=cv2.INTER_LINEAR)

#     _, buffer = cv2.imencode('.png', resized_cutout)
#     mask_data = BytesIO(buffer)
#     cloud_file_path = f"users/users_{int(time.time())}.png"
#     mask_url = convert_to_url(mask_data, cloud_file_path)
#     link = fooocus_endpoint(image_base,mask_url,data.prompt,data.colour)
#     return link


# # @app.post("/colour")
# # async def hair(image_url,colour_image, colour= None):
# #     CLASSES = ['head']
# #     image = read_image_from_url(image_url)
# #     # image = cv2.imread(image_path)
# #     image = preprocess_image_for_opencv(image)
# #     detections = grounding_dino_model.predict_with_classes(
# #         image=image,
# #         classes=enhance_class_name(class_names=CLASSES),
# #         box_threshold=BOX_TRESHOLD,
# #         text_threshold=TEXT_TRESHOLD
# #     )
# #     for x1, y1, x2, y2 in detections.xyxy:
# #      y2 = y2 + 60
# #      x1 = x1 - 60
# #      x2 = x2 + 60
# #      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# #      width = x2 - x1
# #      height = y2 - y1

# #      if width > height:
# #         diff = (width - height) // 2
# #         # y1 = max(y1 - diff, 0)
# #         y2 = min(y2 + diff, image.shape[0])
# #      elif height > width:
# #         diff = (height - width) // 2
# #         x1 = max(x1 - diff, 0)
# #         x2 = min(x2 + diff, image.shape[1])

# #     # Recalculate width and height to verify the square
# #      width = x2 - x1
# #      height = y2 - y1

# #     # Adjust for any remaining one-pixel offset due to integer division
# #      if width > height:
# #         y2 += 1
# #      elif height > width:
# #         x2 += 1
# #      cutout = image[y1:y2, x1:x2]
# #      resized_cutout = cv2.resize(cutout, (1024, 1024), interpolation=cv2.INTER_LINEAR)

# #     _, buffer = cv2.imencode('.png', resized_cutout)
# #     cutout_data = BytesIO(buffer)
# #     cloud_file_path = f"user/user_{int(time.time())}.png"
# #     image_url = convert_to_url(cutout_data, cloud_file_path)
# #     url = "http://localhost:8000/hair_colour"
# #     params = {
# #      "face_path": f"{image_url}",
# #      "shape_path": f"{image_url}",
# #      "color_path": f"{colour_image}"
# #     }
# #     response = requests.post(url, headers=headers, params=params)
# #     # print(response)
# #     print(image_url)
# #     link = fooocus_endpoint(response.json(),image_url)
# #     return link

# # @app.post("/long_hair")
# # async def bald(image_url,colour_image,colour=None):
# #     CLASSES = ['head']
# #     image = read_image_from_url(image_url)
# #     # image = cv2.imread(image_path)
# #     image = preprocess_image_for_opencv(image)
# #     detections = grounding_dino_model.predict_with_classes(
# #         image=image,
# #         classes=enhance_class_name(class_names=CLASSES),
# #         box_threshold=BOX_TRESHOLD,
# #         text_threshold=TEXT_TRESHOLD
# #     )
# #     for x1, y1, x2, y2 in detections.xyxy:
# #      y2 = y2 + 80
# #      x1 = x1 - 80
# #      x2 = x2 + 80
# #      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# #      width = x2 - x1
# #      height = y2 - y1

# #      if width > height:
# #         diff = (width - height) // 2
# #         # y1 = max(y1 - diff, 0)
# #         y2 = min(y2 + diff, image.shape[0])
# #      elif height > width:
# #         diff = (height - width) // 2
# #         x1 = max(x1 - diff, 0)
# #         x2 = min(x2 + diff, image.shape[1])

# #     # Recalculate width and height to verify the square
# #      width = x2 - x1
# #      height = y2 - y1

# #     # Adjust for any remaining one-pixel offset due to integer division
# #      if width > height:
# #         y2 += 1
# #      elif height > width:
# #         x2 += 1
# #      cutout = image[y1:y2, x1:x2]
# #      resized_cutout = cv2.resize(cutout, (1024, 1024), interpolation=cv2.INTER_LINEAR)

# #     _, buffer = cv2.imencode('.png', resized_cutout)
# #     cutout_data = BytesIO(buffer)
# #     cloud_file_path = f"user/user_{int(time.time())}.png"
# #     image_url = convert_to_url(cutout_data, cloud_file_path)
# #     url = "http://localhost:8000/hair_colour"
# #     if colour:
# #      params = {
# #       "face_path": f"{image_url}",
# #       "shape_path": f"{colour_image}",
# #       "color_path": f"{colour_image}"
# #      }
# #     else:
# #      params = {
# #       "face_path": f"{image_url}",
# #       "shape_path": f"{colour_image}",
# #       "color_path": f"{image_url}"
# #      }
# #     image_url = requests.post(url, headers=headers, params=params)
# #     # link = fooocus_endpoint(image_url.json())
# #     return image_url.json()

# def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
#     sam_predictor.set_image(image)
#     result_masks = []
#     for box in xyxy:
#         masks, scores, logits = sam_predictor.predict(
#             box=box,
#             multimask_output=True
#         )
#         index = np.argmax(scores)
#         result_masks.append(masks[index])
#     return np.array(result_masks)


# @app.post("/hair")
# async def hair(data: ImageData):
#         CLASSES = ['hair']
#         BOX_TRESHOLD = 0.40
#         TEXT_TRESHOLD = 0.25
#         image_base = read_image_from_base64(data.image_base64)
#         image = preprocess_image_for_opencv(image_base)
#         detections = grounding_dino_model.predict_with_classes(
#             image=image,
#             classes=enhance_class_name(class_names=CLASSES),
#             box_threshold=BOX_TRESHOLD,
#             text_threshold=TEXT_TRESHOLD
#         )
#         print("yes")
#         detections.mask = segment(
#             sam_predictor=sam_predictor,
#             image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#             xyxy=detections.xyxy
#         )
#         print("yes")
#         mask = detections.mask[0]
#         mask = (mask.astype(np.uint8) * 255)
#         _, buffer = cv2.imencode('.png', mask)
#         mask_data = BytesIO(buffer)
#         cloud_file_path = f"masks/mask_{int(time.time())}.png"
#         mask_url = convert_to_url(mask_data, cloud_file_path)
#         link = fooocus_endpoint(image_base,mask_url,data.prompt,data.colour)
#         return link


# @app.post("/volume")
# async def hair(data: ImageData):
#         CLASSES = ['hair']
#         BOX_TRESHOLD = 0.40
#         TEXT_TRESHOLD = 0.25
#         image_base = read_image_from_base64(data.image_base64)
#         image = preprocess_image_for_opencv(image_base)
#         detections = grounding_dino_model.predict_with_classes(
#             image=image,
#             classes=enhance_class_name(class_names=CLASSES),
#             box_threshold=BOX_TRESHOLD,
#             text_threshold=TEXT_TRESHOLD
#         )
#         detections.mask = segment(
#             sam_predictor=sam_predictor,
#             image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#             xyxy=detections.xyxy
#         )
#         mask = detections.mask[0]
#         mask = (mask.astype(np.uint8) * 255)
#         _, buffer = cv2.imencode('.png', mask)
#         mask_data = BytesIO(buffer)
#         cloud_file_path = f"masks/mask_{int(time.time())}.png"
#         mask_url = convert_to_url(mask_data, cloud_file_path)
#         prompt = "high volume hair, beautiful hair"
#         link = volume_endpoint(image_base,mask_url,data.prompt)
#         return link


# @app.post("/men-long-hair")
# async def hair(data: ImageData):
#     # try:
#         CLASSES = ['hair']
#         BOX_TRESHOLD = 0.40
#         TEXT_TRESHOLD = 0.25
#         image_base = read_image_from_base64(data.image_base64)
#         image = preprocess_image_for_opencv(image_base)
#         detections = grounding_dino_model.predict_with_classes(
#             image=image,
#             classes=enhance_class_name(class_names=CLASSES),
#             box_threshold=BOX_TRESHOLD,
#             text_threshold=TEXT_TRESHOLD
#         )
#         detections.mask = segment(
#             sam_predictor=sam_predictor,
#             image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#             xyxy=detections.xyxy
#         )
#         mask = detections.mask[0]
#         mask = (mask.astype(np.uint8) * 255)
#         CLASSES = ['person']
#         BOX_TRESHOLD = 0.40
#         TEXT_TRESHOLD = 0.25
#         # image_base = read_image_from_base64(data.image_base64)
#         # image = preprocess_image_for_opencv(image)
#         detections = grounding_dino_model.predict_with_classes(
#             image=image,
#             classes=enhance_class_name(class_names=CLASSES),
#             box_threshold=BOX_TRESHOLD,
#             text_threshold=TEXT_TRESHOLD
#         )
#         detections.mask = segment(
#             sam_predictor=sam_predictor,
#             image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#             xyxy=detections.xyxy
#         )
#         mask1 = detections.mask[0]
#         mask1 = (mask1.astype(np.uint8) * 255)
#         mask1 = cv2.bitwise_not(mask1)
#         mask = cv2.bitwise_or(mask,mask1)
#         _, buffer = cv2.imencode('.png', mask)
#         mask_data = BytesIO(buffer)
#         cloud_file_path = f"masks/mask_{int(time.time())}.png"
#         mask_url = convert_to_url(mask_data, cloud_file_path)
#         # prompt = prompt + ", hair behind back,"
#         link = fooocus_endpoint(data.image_base64,mask_url,data.prompt,data.colour)
#         return link
#     # except:
#     #     return {"error" : "hair not detected, try bald option"}

# @app.post("/wig-long")
# async def hair(data: ImageData):
#     image_base = read_image_from_base64(data.image_base64)
#     image = preprocess_image_for_opencv(image_base)
#     CLASSES = ['bald', 'person']
#     BOX_THRESHOLD = 0.40
#     TEXT_THRESHOLD = 0.25

#     # Detect bald areas
#     bald_detections = grounding_dino_model.predict_with_classes(
#         image=image,
#         classes=enhance_class_name(class_names=['bald']),
#         box_threshold=BOX_THRESHOLD,
#         text_threshold=TEXT_THRESHOLD
#     )
#     mask = np.zeros_like(image, dtype=np.uint8)
#     for x1, y1, x2, y2 in bald_detections.xyxy:
#      y1 = y1 - 20
#      y2 = y2 - 30
#      x1 = x1 - 10
#      x2 = x2 + 10
#      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#      mask = np.zeros_like(image, dtype=np.uint8)
#      cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=cv2.FILLED)

#     person_detections = grounding_dino_model.predict_with_classes(
#         image=image,
#         classes=enhance_class_name(class_names=['person']),
#         box_threshold=BOX_THRESHOLD,
#         text_threshold=TEXT_THRESHOLD
#     )
#     mask1 = segment(
#         sam_predictor=sam_predictor,
#         image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
#         xyxy=person_detections.xyxy
#     )[0] * 255
#     mask1 = cv2.bitwise_not(mask1.astype(np.uint8))

#     # Resize mask1 to match image if necessary
#     if mask.shape != mask1.shape:
#         mask1 = cv2.resize(mask1, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
#     if mask1.ndim == 2:
#         mask1 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR) 
#     mask1 = mask1.astype(mask.dtype)
#     try:
#      combined_mask = cv2.bitwise_or(mask, mask1)
#     except:
#         return None 
#     _, buffer = cv2.imencode('.png', combined_mask)
#     mask_data = BytesIO(buffer)
#     cloud_file_path = f"masks/mask_{int(time.time())}.png"
#     mask_url = convert_to_url(mask_data, cloud_file_path)
#     # prompt = prompt + ", hair behind back,"
#     link = fooocus_endpoint(image_base, mask_url, data.prompt, data.colour)
#     return link


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=6000)
