from src.training.trainer_builder import build_trainer
from src.utils.utils import LogManager

def training(param: dict):
    with LogManager(param):
        trainer = build_trainer(param)
        resume_from_checkpoint = param.get('resume_from_checkpoint')
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
