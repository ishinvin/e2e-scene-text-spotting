import os
import json
import torch
import logging
import pathlib
import traceback
import argparse

from modules.utils.util import make_dir

from modules.utils.util import predict
from modules.models.model import OCRModel

logging.basicConfig(level=logging.DEBUG, format='')


def load_model(model_path, with_gpu):
    config = json.load(open('config.json'))
    logger.info("Loading checkpoint: {} ...".format(model_path))
    checkpoints = torch.load(model_path, map_location='cpu')
    if not checkpoints:
        raise RuntimeError('No checkpoint found.')
    print('Epochs: {}'.format(checkpoints['epoch']))
    state_dict = checkpoints['state_dict']
    model = OCRModel(config)
    if with_gpu and torch.cuda.device_count() > 1:
        model.parallelize()
    model.load_state_dict(state_dict)
    if with_gpu:
        model.to(torch.device('cuda'))
    model.eval()
    return model


def main(args: argparse.Namespace):
    model_path = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir
    with_image = True if output_dir else False
    with_gpu = True if torch.cuda.is_available() else False
    # with_gpu = False
    if with_image:
        make_dir(os.path.join(output_dir, 'img'))

    model = load_model(model_path, with_gpu)

    types = ('*.jpg', '*.png', '*.JPG', '*.PNG')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(input_dir.glob(files))

    for image_fn in files_grabbed:
        try:
            with torch.no_grad():
                ploy, im = predict(image_fn, model, with_image, output_dir, with_gpu)
                print(image_fn, len(ploy))
        except Exception as e:
            traceback.print_exc()
            print(image_fn)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model', default=None, type=pathlib.Path, required=True, help='path to model')
    parser.add_argument('-o', '--output_dir', default=None, type=pathlib.Path, help='output dir for drawn images')
    parser.add_argument('-i', '--input_dir', default=None, type=pathlib.Path, required=True, help='dir for input image')
    args = parser.parse_args()
    main(args)
