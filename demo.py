import cv2
import json
import argparse
import torch
import numpy as np

from modules.models.model import OCRModel
from modules.utils.util import converter, sort_poly, show_box


def load_model(model_path, with_gpu):
    config = json.load(open('config.json'))
    checkpoints = torch.load(model_path, map_location='cpu')
    if not checkpoints:
        raise RuntimeError('No checkpoint found.')
    state_dict = checkpoints['state_dict']
    model = OCRModel(config)
    if with_gpu and torch.cuda.device_count() > 1:
        model.parallelize()
    model.load_state_dict(state_dict)
    if with_gpu:
        model.to(torch.device('cuda'))
    model.eval()
    return model


def resize_image(im, max_size=1585152, scale_up=True):
    if scale_up:
        image_size = [im.shape[1] * 3 // 32 * 32, im.shape[0] * 3 // 32 * 32]
    else:
        image_size = [im.shape[1] // 32 * 32, im.shape[0] // 32 * 32]
    while image_size[0] * image_size[1] > max_size:
        image_size[0] /= 1.2
        image_size[1] /= 1.2
        image_size[0] = int(image_size[0] // 32) * 32
        image_size[1] = int(image_size[1] // 32) * 32

    resize_h = int(image_size[1])
    resize_w = int(image_size[0])

    scaled = cv2.resize(im, dsize=(resize_w, resize_h))
    return scaled, (resize_h, resize_w)


def demo(img, model, with_gpu=False):
    im_resized, (ratio_h, ratio_w) = resize_image(img, scale_up=False)
    im_resized = im_resized.astype(np.float32)
    im_resized = torch.from_numpy(im_resized)
    if with_gpu:
        im_resized = im_resized.cuda()

    im_resized = im_resized.unsqueeze(0)
    im_resized = im_resized.permute(0, 3, 1, 2)

    score, geometry, preds, boxes, mapping, rois = model.forward(im_resized, None, None)

    if len(boxes) != 0:
        boxes = boxes[:, :8].reshape((-1, 4, 2))

        # decode predicted text
        pred, preds_size = preds
        _, pred = pred.max(2)
        pred = pred.transpose(1, 0).contiguous().view(-1)
        pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
        pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts

        # drawing box and text
        for i, box in enumerate(boxes):
            box = sort_poly(box.astype(np.int32))
            print(pred_transcripts[i])
            img = show_box(img, box, pred_transcripts[i])

    return img


def main(args):
    model_path = args.model
    with_gpu = True if torch.cuda.is_available() else False
    # with_gpu = False
    model = load_model(model_path, with_gpu)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    ret, img = cap.read()

    with torch.no_grad():
        while ret:
            ret, img = cap.read()
            if ret:
                img = demo(img, model, with_gpu)
                cv2.imshow('img', img)
                cv2.waitKey(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model demo')
    parser.add_argument('-m', '--model', default='saved/E2E-STS/model_best.pth.tar', help='path to model')
    args = parser.parse_args()
    main(args)
