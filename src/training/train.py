from src.training.trainer_builder import build_trainer
from src.utils.utils import LogManager

def training(param: dict):
    log_manager = LogManager(param)
    log_manager.start_logging()

    trainer = build_trainer(param)
    resume_from_checkpoint = param.get('resume_from_checkpoint')
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    log_manager.end_logging()
