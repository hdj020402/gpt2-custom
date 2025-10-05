from training.trainer_builder import build_trainer
from utils.utils import LogManager

def training(param: dict):
    log_manager = LogManager(param)
    log_manager.start_logging()

    trainer = build_trainer(param)
    trainer.train()

    log_manager.end_logging()
