import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from modules.base.base_model import BaseModel
from modules.models.core.crnn import CRNN
from modules.models.core.fpn_resnet import ResNetBackbone
from modules.utils.converter import keys
from modules.utils.util import detect
from modules.utils.roi import batch_roi_transform


class OCRModel:

    def __init__(self, config):
        num_class = len(keys) + 1
        backbone_channel_out = 256
        self.backbone = ResNetBackbone(config)
        self.detector = Detector(config, backbone_channel_out)
        self.recognizer = Recognizer(num_class, config)

    def parallelize(self):
        self.backbone = torch.nn.DataParallel(self.backbone)
        self.recognizer = torch.nn.DataParallel(self.recognizer)
        self.detector = torch.nn.DataParallel(self.detector)

    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.detector = self.detector.to(device)
        self.recognizer = self.recognizer.to(device)

    def summary(self):
        self.backbone.summary()
        self.detector.summary()
        self.recognizer.summary()

    def optimize(self, optimizer_type, params):
        optimizer = getattr(optim, optimizer_type)(
            [
                {'params': self.backbone.parameters()},
                {'params': self.detector.parameters()},
                {'params': self.recognizer.parameters()},
            ],
            **params
        )
        return optimizer

    def train(self):
        self.backbone.train()
        self.detector.train()
        self.recognizer.train()

    def eval(self):
        self.backbone.eval()
        self.detector.eval()
        self.recognizer.eval()

    def state_dict(self):
        sd = {
            '0': self.backbone.state_dict(),
            '1': self.detector.state_dict(),
            '2': self.recognizer.state_dict()
        }
        return sd

    def load_state_dict(self, sd):
        self.backbone.load_state_dict(sd['0'])
        self.detector.load_state_dict(sd['1'])
        self.recognizer.load_state_dict(sd['2'])

    @property
    def training(self):
        return self.backbone.training and self.detector.training and self.recognizer.training

    def forward(self, *inputs):
        """

        :param inputs:
        :return:
        """
        image, boxes, mapping = inputs

        if image.is_cuda:
            device = image.get_device()
        else:
            device = torch.device('cpu')

        feature_map = self.backbone.forward(image)
        score_map, geo_map = self.detector(feature_map)

        if self.training:
            rois = batch_roi_transform(image, boxes[:, :8], mapping)
            pred_mapping = mapping
            pred_boxes = boxes
        else:
            score = score_map.permute(0, 2, 3, 1)
            geometry = geo_map.permute(0, 2, 3, 1)
            score = score.detach().cpu().numpy()
            geometry = geometry.detach().cpu().numpy()

            pred_boxes = []
            pred_mapping = []
            for i in range(score.shape[0]):
                s = score[i, :, :, 0]
                g = geometry[i, :, :, ]
                bb = detect(score_map=s, geo_map=g)
                bb_size = bb.shape[0]

                if len(bb) > 0:
                    pred_mapping.append(np.array([i] * bb_size))
                    pred_boxes.append(bb)

            if len(pred_mapping) > 0:
                pred_boxes = np.concatenate(pred_boxes)
                pred_mapping = np.concatenate(pred_mapping)
                rois = batch_roi_transform(image, pred_boxes[:, :8], pred_mapping)
            else:
                return score_map, geo_map, (None, None), pred_boxes, pred_mapping, None

        preds = self.recognizer(rois)
        preds_size = torch.LongTensor([preds.size(0)] * int(preds.size(1))).to(device)

        return score_map, geo_map, (preds, preds_size), pred_boxes, pred_mapping, rois


class Recognizer(BaseModel):

    def __init__(self, nclass, config):
        super().__init__(config)
        self.crnn = CRNN(32, 1, nclass, 256)
        # self.crnn.register_backward_hook(self.crnn.backward_hook)

    def forward(self, rois):
        return self.crnn(rois)


class Detector(BaseModel):

    def __init__(self, config, backbone_channel):
        super().__init__(config)
        self.score_map = nn.Conv2d(backbone_channel, 1, kernel_size=1)
        self.geo_map = nn.Conv2d(backbone_channel, 4, kernel_size=1)
        self.angle_map = nn.Conv2d(backbone_channel, 1, kernel_size=1)
        self.scale = config['data_loader']['input_size']

    def forward(self, inputs):
        score = torch.sigmoid(self.score_map(inputs))

        # 出来的是 normalise 到 0 -1 的值是到上下左右的距离，但是图像他都缩放到  512 * 512 了，但是 gt 里是算的绝对数值来的
        geoMap = torch.sigmoid(self.geo_map(inputs)) * self.scale
        angleMap = (torch.sigmoid(self.angle_map(inputs)) - 0.5) * math.pi / 2
        geometry = torch.cat([geoMap, angleMap], dim=1)

        return score, geometry
