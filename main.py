import os
import logging
import wandb

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from src.inference import Inference
from src.trainer import Trainer


def init_logger():
    """
    날짜, 시간을 logging 해주는 함수
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)


def make_config(cfg: DictConfig):
    result = {}
    result.update(dict(cfg.data))
    result.update(dict(cfg.model))
    result.update(dict(cfg.trainer))
    return result


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    cfg.data.train_path = os.path.join(get_original_cwd(), cfg.data.train_path)
    cfg.data.test_path = os.path.join(get_original_cwd(), cfg.data.test_path)
    cfg.data.save_path = os.path.join(get_original_cwd(), cfg.data.save_path)

    wandb.init(project="Conference_Transcript_Summary")
    wandb.run.name = cfg.project_name
    wandb.config.update(make_config(cfg))
    trainer = Trainer(cfg)
    wandb.watch(trainer.model.model)
    init_logger()
    trainer.forward()


@hydra.main(config_path="config", config_name="config")
def test(cfg: DictConfig) -> None:
    cfg = cfg.inference
    cfg.model_path = os.path.join(get_original_cwd(), cfg.model_path)
    cfg.test_path = os.path.join(get_original_cwd(), cfg.test_path)
    cfg.submission_path = os.path.join(get_original_cwd(), cfg.submission_path)
    cfg.save_path = os.path.join(get_original_cwd(), cfg.save_path)

    tester = Inference(cfg)
    tester.forward()


if __name__ == '__main__':
    train()
