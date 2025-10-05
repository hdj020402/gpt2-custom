import optuna
from typing import Callable

from training.trainer_builder import build_trainer
from utils.utils import LogManager

def make_hp_space(ht_param: dict[str, dict]) -> Callable[[optuna.Trial], dict]:
    def hp_space(trial: optuna.Trial) -> dict:
        SUGGEST_METHOD_MAP = {
            'int': trial.suggest_int,
            'float': trial.suggest_float,
            'uniform': trial.suggest_uniform,
            'discrete_uniform': trial.suggest_discrete_uniform,
            'loguniform': trial.suggest_loguniform,
            'categorical': trial.suggest_categorical,
        }
        space = {}
        for hparam, attr in ht_param.items():
            if hparam == 'optuna':
                continue
            suggest_method = SUGGEST_METHOD_MAP[attr['type']]
            sm_kwargs = {k: v for k, v in attr.items() if k != 'type'}
            space[hparam] = suggest_method(hparam, **sm_kwargs)
        return space
    return hp_space

def hpo(param: dict, ht_param: dict):
    log_manager = LogManager(param)
    log_manager.start_logging()

    trainer = build_trainer(param)
    best_run = trainer.hyperparameter_search(
        hp_space=make_hp_space(ht_param),
        backend="optuna",
        direction=ht_param['optuna']['direction'],
        n_trials=ht_param['optuna']['n_trials'],
        study_name=f"hptuning_{param['jobtype']}",
        storage=log_manager.optuna_db,
    )

    log_manager.end_logging()

    return best_run
