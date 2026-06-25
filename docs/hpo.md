# HPO Reference

This page lists the optuna sampler / pruner classes available in `configs/hpo.yml`.
Set `type` to any class name below and add its constructor parameters as YAML keys.

---

## Samplers

Samplers decide *how* the next trial's hyperparameters are chosen.

### TPESampler (recommended default)

Tree-structured Parzen Estimator — models promising regions and samples from them.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `seed` | `int` | `None` | Reproducibility |
| `n_startup_trials` | `int` | `10` | Random sampling before TPE kicks in |
| `n_ei_candidates` | `int` | `24` | Candidates per sampling |
| `multivariate` | `bool` | `False` | Model parameter interactions |
| `constant_liar` | `bool` | `False` | Use constant-liar for parallel HPO |

Example:
```yaml
sampler:
  type: TPESampler
  seed: 42
  n_startup_trials: 10
```

### GridSampler

Exhaustive search over all combinations. **Every hyperparameter must use `type: categorical`.**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `seed` | `int` | `None` | Shuffles the grid order |
| `search_space` | `dict` | `None` | Usually omitted — inferred from `hp_space` |

Example:
```yaml
sampler:
  type: GridSampler
  seed: 42
```

### RandomSampler

Uniform random sampling from the search space.

| Parameter | Type | Default |
|-----------|------|---------|
| `seed` | `int` | `None` |

### CmaEsSampler

Covariance Matrix Adaptation Evolution Strategy — population-based, good for continuous
spaces where parameters interact.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `seed` | `int` | `None` | |
| `n_startup_trials` | `int` | `1` | |
| `popsize` | `int` | `None` | Population size (auto if unset) |
| `restart_strategy` | `str` | `None` | `"ipop"` or `"bipop"` |
| `consider_pruned_trials` | `bool` | `False` | Include pruned trials in CMA estimation |

### QMCSampler

Quasi-Monte Carlo — low-discrepancy sequences for even coverage.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `seed` | `int` | `None` | |
| `qmc_type` | `str` | `"sobol"` | `"sobol"` or `"halton"` |
| `scramble` | `bool` | `False` | Randomize sequence (better for high-dim) |

---

## Pruners

Pruners stop unpromising trials early.

### MedianPruner (recommended default)

Prune if the current value is worse than the median **at the same step** across trials.
Safe and simple.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `n_startup_trials` | `int` | `5` | Don't prune before N completed trials |
| `n_warmup_steps` | `int` | `0` | Don't prune before step N in each trial |
| `interval_steps` | `int` | `1` | Check pruning every N steps |
| `n_min_trials` | `int` | `1` | Min trials at a step before pruning there |

Example:
```yaml
pruner:
  type: MedianPruner
  n_startup_trials: 5
  n_warmup_steps: 20
```

### HyperbandPruner

Multi-fidelity pruner — runs many configurations with few resources, progressively
allocating more budget to the best.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `min_resource` | `int` | `1` | Minimum steps before first pruning |
| `max_resource` | `int/str` | `"auto"` | Maximum steps (auto = full training) |
| `reduction_factor` | `int` | `3` | Keep 1/N each round (higher = more aggressive) |
| `bootstrap_count` | `int` | `0` | Bootstrap samples for uncertainty |

Example:
```yaml
pruner:
  type: HyperbandPruner
  min_resource: 100
  reduction_factor: 3
```

### SuccessiveHalvingPruner

Simpler version of Hyperband — fixed budget allocation.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `min_resource` | `int/str` | `"auto"` | Minimum steps |
| `reduction_factor` | `int` | `4` | Keep 1/N each round |
| `min_early_stopping_rate` | `int` | `0` | Minimum halving rounds |
| `bootstrap_count` | `int` | `0` | Bootstrap uncertainty estimate |

### PercentilePruner

Prune trials in the bottom K% at each step.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `percentile` | `float` | *required* | Bottom percentile to prune (e.g. `25`) |
| `n_startup_trials` | `int` | `5` | |
| `n_warmup_steps` | `int` | `0` | |
| `interval_steps` | `int` | `1` | |

### ThresholdPruner

Prune when value crosses a fixed threshold.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `lower` | `float` | `None` | Prune if value < lower (for `direction: minimize`) |
| `upper` | `float` | `None` | Prune if value > upper (for `direction: maximize`) |
| `n_warmup_steps` | `int` | `0` | |
| `interval_steps` | `int` | `1` | |

### PatientPruner

Wraps another pruner — only prunes after `patience` consecutive unimproved steps.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `wrapped_pruner` | `optuna.pruner` | *required* | The underlying pruner |
| `patience` | `int` | *required* | Consecutive steps without improvement before pruning |
| `min_delta` | `float` | `0.0` | Minimum change to count as "improvement" |

### WilcoxonPruner

Statistical test — prune if performance is significantly worse than the best trial.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `p_threshold` | `float` | `0.1` | p-value cutoff |
| `n_startup_steps` | `int` | `2` | Wait before testing |

### NopPruner

No pruning. Accepts no parameters.

```yaml
pruner:
  type: NopPruner
```

---

## Quick Reference

### I just want something safe

```yaml
optuna:
  sampler:
    type: TPESampler
    seed: 42
  pruner:
    type: MedianPruner
    n_warmup_steps: 20
  direction: minimize
  n_trials: 100
```

### I have many trials and want to save time

```yaml
sampler:
  type: TPESampler
  seed: 42
pruner:
  type: HyperbandPruner
  min_resource: 100
  reduction_factor: 3
```

### I want to exhaustively test a few values

```yaml
sampler:
  type: GridSampler
# No pruner needed — all combinations run completely
```
