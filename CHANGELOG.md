# Changelog

## Branch — parallel-training (2026-05-05)

### Design Changes

**`training/optuna_search.py` — switched from SQLite to JournalStorage for parallel-safe trial storage**
SQLite serializes all writes through a single file lock. When multiple Optuna workers try to record a trial result simultaneously, SQLite raises a locking error and the trial is lost. `JournalStorage` with `JournalFileBackend` uses append-only file writes with OS-level locking — multiple processes can write concurrently without conflicts. No database server required; the journal is a plain `.log` file in `optuna_studies/`.

**`training/optuna_search.py` — MLflow experiment pre-created before workers start**
When `n_jobs > 1`, Optuna spawns worker processes that each call `setup_mlflow()` independently. If the experiment does not exist yet, multiple workers can race to create it simultaneously, resulting in duplicate experiments or write errors. Fixed by calling `setup_mlflow()` once in the main process inside `run_study()` before `study.optimize()` launches workers. Worker calls to `setup_mlflow()` are then idempotent — the experiment already exists and they just set it as active.

**`training/optuna_search.py` + `main.py` — `n_jobs` parameter added**
`run_study()` now accepts `n_jobs` which is passed to `study.optimize(n_jobs=n_jobs)`. Defaults to `N_JOBS` from config (set to 4). Overridable at the command line with `--n-jobs`. Progress bar is automatically disabled when `n_jobs > 1` because Optuna's tqdm bar is not thread-safe with parallel workers.

**`config.py` — `N_JOBS = 4` added**
Default parallel trial count. On a laptop set this to 1. On the Lambda server set it to the number of available CPU cores (check with `nproc`).

---



## Session — 2026-05-04 (continued)

### Performance Fix

**`agents/buffer.py` — `ReplayBuffer` and `SequenceReplayBuffer` rewritten to use pre-allocated structures**
Both buffers had the same O(n) sampling bottleneck. `ReplayBuffer` used `random.sample(deque, batch_size)` which is O(n) because Python must convert the deque to a list before sampling. `SequenceReplayBuffer` used `random.choice(deque)` which is O(n) for the same reason — deques have no random access. Both replaced with pre-allocated structures (numpy arrays for `ReplayBuffer`, a fixed-size list for `SequenceReplayBuffer`) and circular pointer logic. Sampling is now O(batch_size) in both cases. Interface, tensor shapes, and episode-based logic are unchanged.

**`agents/buffer.py` — `ReplayBuffer` rewritten to use pre-allocated numpy arrays (detail)**
The previous implementation stored transitions as Python tuples in a `deque`. `random.sample()` on a `deque` is O(n) because Python must convert the entire deque to a list before picking random indices — at 25,000 capacity and training every step, this was the primary wall-clock bottleneck. Replaced with pre-allocated numpy arrays `(capacity, obs_dim)` and a circular buffer with a `pos` pointer. Sampling is now `np.random.randint(0, upper, size=batch_size)` followed by direct array indexing — O(batch_size) instead of O(capacity). Interface (`push`, `sample`, `is_ready`, `__len__`) and returned tensor shapes are unchanged. The only behavioral difference is sampling with replacement instead of without, which is negligible at our buffer sizes (~1% chance of duplicate at batch=256, buffer=25k). `SequenceReplayBuffer` (DRQN) is unchanged — its episode-based structure is inherently different and does not have the same bottleneck.

---

## Session — 2026-05-04

### Design Changes

**`config.py` — `N_TRIALS` reduced from 50 to 25**
Optuna's TPE sampler uses the first 10 trials for random exploration before its surrogate model activates, leaving 15 TPE-guided trials. For a 6–7 dimensional search space this is sufficient to identify a strong region without exhaustive coverage — TPE has strong diminishing returns past ~20 trials at this scale. Reduced for practical runtime reasons; defensible because TPE is more sample-efficient than grid or random search at this dimensionality.

**`config.py` — `total_timesteps` raised from 10,000 to 75,000**
The Optuna objective (`learning_speed_score`) returns 0 whenever no level is cleared. At 10,000 steps even well-tuned hyperparameters cannot clear a level, so the vast majority of trials returned 0 and Optuna's TPE sampler had a flat landscape to learn from — effectively making the search random. 75,000 steps is the smallest budget where a fast-learning agent can realistically clear multiple levels, making the objective discriminative: good hyperparameters score high, bad ones score 0. Any larger budget risks measuring final convergence rather than learning speed, which contradicts the goal.

**`config.py` — `epsilon_decay` changed from 0.99 to 0.99994**
`epsilon_decay` is applied every step, not every episode. At 0.99, epsilon reached its floor of 0.05 after approximately 298 steps — meaning the agent stopped exploring after less than 0.4% of the training budget and spent the remaining 99.6% exploiting an essentially random policy. At 0.99994 with a 75,000-step budget, epsilon reaches 0.05 at around step 50,000 (67% of training), giving the agent meaningful exploration across the majority of the run before committing to exploitation. These two parameters (total_timesteps and epsilon_decay) are coupled: changing the budget without recalibrating the decay — or vice versa — would reintroduce the same collapse.

**`config.py` — `buffer_size` upper bound reduced from 50,000 to 25,000**
With a training budget of 75,000 steps, any buffer larger than 75,000 can never be filled. The previous upper bound of 50,000 meant Optuna was exploring values that were functionally identical to a full buffer in the upper half of the range — wasting trials on a parameter with no observable effect. The new ceiling of 25,000 (one-third of the total budget) ensures every value in the search space is reachable and distinct, so Optuna's model of the landscape is accurate.

### Bug Fixes

**`training/train.py` — tracker level-clear threshold mismatched env advancement threshold**
`ProgressTracker.log_episode` used a tolerance of `6 * len(pattern_list)` to decide when a level was "cleared", but `env.reset()` advances the agent to the next level using a tolerance of `15 * len(pattern_list)` (for levels 1–5). This meant an agent could complete all 6 levels by the environment's standard, trigger the `curr_wave → 1` rollback, and the break condition (`if ENV_CONFIG["levels"] in tracker.level_clear_steps`) would never fire because the tracker had never recorded level 6 as cleared. Training would continue past the reset, mixing level-1 observations into what should be level-6 data. Fixed by aligning the tracker threshold to `15 * len(pattern_list)` so "cleared" in the tracker means the same thing as "advanced" in the environment.

**`training/train.py` — `_dqn_train_step` missing gradient clipping**
`_drqn_train_step` already applied `clip_grad_norm_(max_norm=10)` to prevent exploding gradients, but `_dqn_train_step` did not. Early in training, when Q-values are uninitialised and rewards are sparse, large Bellman errors produce large loss values that can cause unbounded gradient updates and destabilise learning. Added `clip_grad_norm_` to DQN to match DRQN and ensure consistent gradient behaviour across both agents.

---

## Session — 2026-04-28

### Bug Fixes

**`training/train.py` — `ProgressTracker.steps` never populated**
`self.steps` was initialised as an empty list but `log_episode` never appended to it. Added `self.steps.append(step)` so the list correctly tracks the global step number at each episode boundary.

**`training/train.py` — `obs_dim` and `action_dim` hardcoded**
Both `train_dqn` and `train_drqn` had `obs_dim=72, action_dim=7` written as magic numbers. These are now derived from `ENV_CONFIG["num_vertices"]` so changing `num_vertices` in config propagates correctly to the networks.
```python
nv = ENV_CONFIG["num_vertices"]
obs_dim    = nv * 2 * nv   # grid is (num_vertices × 2*num_vertices)
action_dim = nv + 1        # rotate 0..n-1 + lock action
```

**`env/hexxed.py` — `spawn.txt` loaded via relative path**
`np.loadtxt('spawn.txt', ...)` would crash whenever the process was not started from the project root. Changed to an absolute path derived from `__file__` so the env can be imported from anywhere.

**`tracking/mlflow_logger.py` — `log_trial_result` missing**
`optuna_search.py` imported `log_trial_result` but the function did not exist, causing an `ImportError` on startup. Added the function; it logs the trial objective value as a single MLflow metric.

**`agents/buffer.py` — `SequenceReplayBuffer.is_ready` checked episode count, not transitions**
`len(self)` returned the number of complete *episodes* stored. With `batch_size` up to 256, `is_ready` required 256 full episodes before the first training step — for a 10k-step budget with short episodes, training barely started. Fixed by tracking `_total_transitions` (accounting for deque evictions) and checking that instead.

**`training/train.py` — `log_episode` never called during training**
`mlflow_logger.log_episode` was defined and intended to log per-episode metrics (reward, epsilon, level, loss) but was never called, so MLflow runs had no learning curves — only end-of-run summaries. Added `mlflow_log_episode(...)` calls at the end of each episode in both `train_dqn` and `train_drqn`.

---

### Design Changes

**Objective metric changed to learning speed score**
The previous Optuna objective was `mean(last 20 episode rewards)`, which measures final-state performance and is corrupted if the agent completes all 6 levels and the environment resets to level 1 (harder agents look weaker).

Replaced with a **learning speed score**:
```
score = Σ (total_timesteps − step_when_level_cleared)  for each cleared level
```
Higher = faster generalisation. An agent that never clears a level scores 0. Clearing all levels early in the budget scores highest. This is the primary Optuna objective and is also logged to MLflow as `learning_speed_score`.

**Early stopping when all levels cleared**
Both training loops now `break` as soon as `ENV_CONFIG["levels"]` appears in `tracker.level_clear_steps`. This prevents the environment's level-1 rollback from contaminating any downstream metric.

**`DRQN_SEARCH_SPACE["seq_len"]` upper bound reduced from 32 to 12**
Level 1/2 episodes are typically 2–15 steps long. With `seq_len > episode_length`, `SequenceReplayBuffer` discards every episode and no training occurs. Capped at 12 to keep seq_len within the range where episodes are long enough to produce valid sequences.
