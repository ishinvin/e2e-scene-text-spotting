import os
import json
import logging
import argparse

from modules.models.loss import E2ELoss
from modules.models.model import OCRModel
from modules.logger.logger import Logger
from modules.data.loader import ICDARDataLoader
from modules.trainer.trainer import Trainer
from modules.models.metric import icdar_metric

logging.basicConfig(level=logging.DEBUG, format='')


def main(config, resume):
    train_logger = Logger()

    # load data
    train_dataloader = ICDARDataLoader(config).train()
    val_dataloader = ICDARDataLoader(config).val() if config['validation']['validation_split'] > 0 else None

    # initial model
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    model = OCRModel(config)
    model.summary()

    loss = E2ELoss()
    trainer = Trainer(model, loss, icdar_metric, resume, config, train_dataloader, val_dataloader, train_logger)
    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='MLT-OCR')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args = parser.parse_args()
    config = json.load(open('config.json'))
    if args.resume:
        logger.warning('Warning: --config overridden by --resume')

    main(config, args.resume)
