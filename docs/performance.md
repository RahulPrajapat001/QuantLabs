# Feature build performance

_Generated 2026-07-24 14:22 UTC by `quantlab.feature_engineering.benchmark`._

Target workload: **500 securities x 2520 trading days (74 features)**.

## Optimisation 1 — cache shared intermediates in the DAG

Measured single-threaded on 25 securities. The naive
baseline computes each feature (and its full dependency closure) in isolation, so
shared intermediates such as `ema_12`, `ema_26`, `return_1d` and `log_return_1d`
are recomputed once per dependent feature. The DAG build computes each node once.

| Build | Seconds |
|---|---:|
| Naive (per-feature recompute) | 7.40 |
| DAG-cached | 5.99 |
| **Speed-up** | **1.2x** |

## Optimisation 2 — vectorise rolling operations

Indicators use pandas `ewm`/`rolling` (C-level) instead of Python loops. Measured
on a single long series (EMA span 26).

| EMA implementation | Seconds |
|---|---:|
| Hand-rolled Python loop | 0.0115 |
| Vectorised `ewm` | 0.0008 |
| **Speed-up** | **14.2x** |

## Optimisation 3 — parallelise per-ticker work with joblib

Securities are independent, so the per-security time-series compute fans out
across cores. Measured at the full 500-security scale.

| Build | Seconds |
|---|---:|
| Serial (`n_jobs=1`) | 158.15 |
| Parallel (`n_jobs=-1`) | 49.03 |
| **Speed-up** | **3.2x** |

## Profile (top cumulative time)

Single-threaded cached build; `cProfile`, sorted by cumulative time:

```
46804998 function calls (46055754 primitive calls) in 89.572 seconds

   Ordered by: cumulative time
   List reduced from 1203 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.024    0.024   89.572   89.572 D:\QUANTLABS\src\quantlab\feature_engineering\build.py:136(build_feature_panel)
        1    0.000    0.000   45.355   45.355 D:\QUANTLABS\.venv\Lib\site-packages\joblib\parallel.py:1969(__call__)
      102    0.001    0.000   45.355    0.445 D:\QUANTLABS\.venv\Lib\site-packages\joblib\parallel.py:1888(_get_sequential_output)
      100    0.003    0.000   45.351    0.454 D:\QUANTLABS\src\quantlab\feature_engineering\build.py:121(_compute_one_security)
      100    0.155    0.002   45.129    0.451 D:\QUANTLABS\src\quantlab\feature_engineering\engine.py:30(compute_time_series_features)
        1    0.070    0.070   43.706   43.706 D:\QUANTLABS\src\quantlab\feature_engineering\engine.py:78(compute_cross_sectional_features)
        6    0.464    0.077   43.497    7.249 D:\QUANTLABS\src\quantlab\feature_engineering\engine.py:56(_apply_cross_sectional)
     6500    0.032    0.000   22.616    0.003 D:\QUANTLABS\.venv\Lib\site-packages\pandas\core\window\rolling.py:538(_apply)
     6500    0.047    0.000   22.543    0.003 D:\QUANTLABS\.venv\Lib\site-packages\pandas\core\window\rolling.py:444(_apply_columnwise)
     6500    0.094    0.000   22.429    0.003 D:\QUANTLABS\.venv\Lib\site-packages\pandas\core\window\rolling.py:424(_apply_series)
     6500    0.073    0.000   21.119    0.003 D:\QUANTLABS\.venv\Lib\site-packages\pandas\core\window\rolling.py:571(homogeneous_func)
     6500    0.440    0.000   20.966    0.003 D:\QUANTLABS\.venv\Lib\site-packages\pandas\core\window\rolling.py:577(calc)
      600    0.003    0.000   20.181    0.034 D:\QUANTLABS\.venv\Lib\site-packages\pandas\core\window\rolling.py:2167(apply)
      600    0.006    0.000   20.178    0.034 D:\QUANTLABS\.venv\Lib\site-packages\pandas\core\window\rolling.py:1526(apply)
    15120    0.225    0.000   20.025    0.001 D:\QUANTLABS\.venv\Lib\site-packages\pandas\core\indexing.py:917(__setitem__)
```
