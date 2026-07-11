import glob
import inspect
import json
import math
import optuna
import logging
import shutil
import torch
from typing import Callable

from transformers import TrainerCallback

from src.training.trainer_builder import build_trainer
from src.utils.utils import LogManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mappings from YAML string → optuna class
# ---------------------------------------------------------------------------
SAMPLER_MAP: dict[str, type] = {
    name: getattr(optuna.samplers, name)
    for name in (
        'TPESampler', 'GridSampler', 'RandomSampler', 'CmaEsSampler',
        'GPSampler', 'QMCSampler', 'NSGAIISampler', 'NSGAIIISampler',
        'BruteForceSampler', 'PartialFixedSampler',
    )
}

PRUNER_MAP: dict[str, type] = {
    name: getattr(optuna.pruners, name)
    for name in (
        'MedianPruner', 'HyperbandPruner', 'ThresholdPruner',
        'SuccessiveHalvingPruner', 'PercentilePruner', 'WilcoxonPruner',
        'NopPruner',
    )
}


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


def _validate_known_params(cls_name: str, cls: type, cfg: dict) -> None:
    """Raise if *cfg* contains keys that don't match the class constructor."""
    sig = inspect.signature(cls.__init__)
    valid = {
        pn for pn, p in sig.parameters.items()
        if pn != 'self'
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                           inspect.Parameter.VAR_KEYWORD)
    }
    unknown = set(cfg.keys()) - valid
    if unknown:
        raise ValueError(
            f"{cls_name} does not accept parameters: {sorted(unknown)}.\n"
            f"Valid parameters are: {sorted(valid) if valid else '(none)'}.\n"
            f"Check your hpo.yml and remove or rename the unknown keys."
        )


def _build_optuna_kwargs(ht_param: dict, param: dict) -> dict:
    """Build sampler, pruner, and extra kwargs for ``hyperparameter_search``.

    Returns a dict with keys: sampler, pruner, and any extra kwargs
    (storage, study_name, load_if_exists, n_trials override).
    """
    opt_cfg = ht_param.get('optuna', {})
    kwargs: dict = {}

    # ── Sampler ──
    sampler_cfg = opt_cfg.get('sampler')
    if sampler_cfg:
        sampler_cfg = dict(sampler_cfg)  # copy to avoid mutating original YAML dict
        sampler_type = sampler_cfg.pop('type')
        sampler_cls = SAMPLER_MAP[sampler_type]
        sampler_seed = sampler_cfg.get('seed')
        if sampler_seed is None and 'seed' in opt_cfg:
            sampler_cfg['seed'] = opt_cfg['seed']
        # GridSampler requires the search space at construction time
        if sampler_type == 'GridSampler':
            sampler_cfg['search_space'] = {
                hp: attr['choices']
                for hp, attr in ht_param.items()
                if hp != 'optuna' and 'choices' in attr
            }
        _validate_known_params(sampler_type, sampler_cls, sampler_cfg)
        kwargs['sampler'] = sampler_cls(**sampler_cfg)

    # ── Pruner ──
    pruner_cfg = opt_cfg.get('pruner')
    if pruner_cfg:
        pruner_cfg = dict(pruner_cfg)  # copy to avoid mutating original YAML dict
        pruner_type = pruner_cfg.pop('type')
        pruner_cls = PRUNER_MAP[pruner_type]
        _validate_known_params(pruner_type, pruner_cls, pruner_cfg)
        kwargs['pruner'] = pruner_cls(**pruner_cfg)

    # ── Continue trials ──
    continue_cfg = opt_cfg.get('continue_trials', {})
    if continue_cfg.get('continue', False):
        storage = continue_cfg.get('storage')
        if storage is None:
            raise ValueError(
                "continue_trials.continue=True requires continue_trials.storage "
                "pointing to the existing Optuna database file."
            )
        kwargs['storage'] = f'sqlite:///{storage}'
        kwargs['study_name'] = continue_cfg.get('study_name') or f"hpo_{param['jobtype']}"
        kwargs['load_if_exists'] = True

    return kwargs


def _validate_gridsampler(ht_param: dict, n_trials: int) -> int:
    """Validate GridSampler constraints and return the effective n_trials.

    For GridSampler: ensure every hyperparameter is categorical, compute the
    total grid size, log a message if *n_trials* differs from the grid size,
    and return the grid size.
    For other samplers: return *n_trials* unchanged.
    """
    opt_cfg = ht_param.get('optuna', {})
    sampler_cfg = opt_cfg.get('sampler', {})
    if sampler_cfg.get('type') != 'GridSampler':
        return n_trials

    grid_size = 1
    bad_params: list[str] = []

    for hparam, attr in ht_param.items():
        if hparam == 'optuna':
            continue
        if attr.get('type') != 'categorical':
            bad_params.append(hparam)
        else:
            grid_size *= len(attr['choices'])

    if bad_params:
        raise ValueError(
            f"GridSampler requires all hyperparameters to use type: categorical.\n"
            f"Non-categorical parameters found: {bad_params}\n"
            f"Either switch to categorical with explicit choices, "
            f"or use a different sampler (e.g. TPESampler)."
        )

    if n_trials != grid_size:
        logger.info(
            f"GridSampler: overriding n_trials {n_trials} → {grid_size} "
            f"({math.prod(len(ht_param[p]['choices'])
             for p in ht_param if p != 'optuna')} combinations)"
        )
        return grid_size
    return n_trials


def _is_rank_zero() -> bool:
    """Return True on rank 0, or in non-distributed mode."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def hpo(param: dict, ht_param: dict):
    opt_cfg = ht_param.get('optuna', {})
    n_trials = _validate_gridsampler(ht_param, opt_cfg.get('n_trials', 20))

    with LogManager(param) as lm:
        hp_kwargs = {
            'storage': lm.optuna_db,
            'study_name': f"hpo_{param['jobtype']}",
        }
        hp_kwargs.update(_build_optuna_kwargs(ht_param, param))

        trainer = build_trainer(param)

        # Trainer.hyperparameter_search returns trainer.objective, which is the
        # *last* eval-step value (not the best within the trial).  This is wrong
        # for Bayesian samplers like TPE that use trial values to model the
        # search space.  Fix: a callback overrides trainer.objective with
        # state.best_metric (the true best across all eval steps) at the end of
        # each trial.
        class _FixObjectiveCallback(TrainerCallback):
            def on_train_end(self, args, state, control, **kwargs):
                if state.best_metric is not None:
                    trainer.objective = state.best_metric

        trainer.add_callback(_FixObjectiveCallback())

        # compute_objective: extract the metric that metric_for_best_model
        # points to, instead of the default (eval_loss or sum-of-all-metrics).
        metric_name = trainer.args.metric_for_best_model
        # Trainer prefixes all eval metrics with "eval_" (see trainer.py line ~2818).
        # Match the logic in trainer._evaluate so the lookup succeeds.
        hp_key = f"eval_{metric_name}" if not metric_name.startswith("eval_") else metric_name
        def compute_objective(metrics: dict) -> float:
            return metrics[hp_key]

        best_run = trainer.hyperparameter_search(
            hp_space=make_hp_space(ht_param),
            backend="optuna",
            direction=opt_cfg['direction'],
            n_trials=n_trials,
            compute_objective=compute_objective,
            **hp_kwargs,
        )

        # In DDP mode, only rank 0 gets a BestRun object; other ranks get None.
        if _is_rank_zero():
            logger.info(best_run)

            best_run_path = f"{param['output_dir']}/best_run.json"
            with open(best_run_path, 'w') as f:
                json.dump({
                    "objective": best_run.objective,
                    "hyperparameters": best_run.hyperparameters,
                }, f, indent=2)
            logger.info(f"Best run (objective={best_run.objective}) saved to {best_run_path}")

            for ckpt_dir in glob.glob(f"{param['output_dir']}/checkpoint-*"):
                try:
                    shutil.rmtree(ckpt_dir)
                    logger.info(f"Cleaned up checkpoint: {ckpt_dir}")
                except FileNotFoundError:
                    pass

    return best_run
