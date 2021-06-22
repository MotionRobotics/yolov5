import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox


class Yolov5:

    @torch.no_grad()
    def __init__(self,
                weights='yolov5s.pt',  # model.pt path(s)
                imgsz=1280,  # inference size (pixels)
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                view_img=False,  # show results
                line_thickness=3,  # bounding box thickness (pixels)
                half_precision=False,  # use FP16 half-precision inference
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                hide_labels=False,  # hide labels
                hide_conf=False,  # hide confidences
                augment=False, # augmented inference
                agnostic_nms=False  # class-agnostic NMS
                ):

        self._imgsz = imgsz
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._max_det = max_det
        self._view_img = view_img
        self._line_thickness = line_thickness
        self._half_precision = half_precision
        self._augment = augment
        self._classes = classes
        self._agnostic_nms = agnostic_nms
        self._hide_labels = hide_labels
        self._hide_conf = hide_conf
        
        # Initialize
        # set_logging()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self._device = select_device(device)
        self._half_precision &= self._device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self._model = attempt_load(weights, map_location=self._device)  # load FP32 model
        stride = int(self._model.stride.max())  # model stride
        imgsz = check_img_size(self._imgsz, s=stride)  # check image size
        # Get class names
        self._names = self._model.module.names if hasattr(self._model, 'module') else self._model.names  # get class names

        if self._half_precision:
            self._model.half()  # to FP16

        # Run inference
        if self._device.type != 'cpu':
            self._model(torch.zeros(1, 3, imgsz, imgsz).to(self._device).type_as(next(self._model.parameters())))  # run once

    @torch.no_grad()
    def __call__(self, cur_img):
        """
        Run inference on img then display result and print bbox info.
        """
        result_img = cur_img.copy()
        ## Reformat image ##
        
        # Padded resize
        img = letterbox(cur_img, self._imgsz, stride=32)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self._device)
        img = img.half() if self._half_precision else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self._model(img, augment=self._augment)[0]

        # Apply Non-Maximum Supression (NMS)
        pred = non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms, max_det=self._max_det)

        for i, det in enumerate(pred):  # detections per image
            
            if len(det):

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self._hide_labels else (self._names[c] if self._hide_conf else f'{self._names[c]} {conf:.2f}')
                    plot_one_box(xyxy, result_img, label=label, color=colors(c, True), line_thickness=self._line_thickness)

        return result_img
