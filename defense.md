# Design Decision Defense

---

## Environment

**1. Hexagon with 6 vertices**
6 vertices gives a non-trivial action space (7 actions) and a 72-dimensional observation without being computationally intractable. Fewer vertices make the task too easy; more increase training time beyond a reasonable budget.

**2. `step_per_pattern = 6`**
Controls how many growth steps of the target pattern the agent can observe. 6 steps gives enough temporal signal for the agent to infer the pattern without making episodes excessively long.

**3. `levels = 6`**
Defines the difficulty ceiling. 6 levels provides a meaningful curriculum — each level adds a new pattern the agent must learn — while keeping the total task within the training budget.

**4. `shuffle_patterns = True`**
Patterns within a level are shuffled each attempt. This prevents the agent from memorizing a fixed sequence and forces it to generalize to the underlying pattern structure rather than rote order.

**5. `random_rolls = False`**
Rotational augmentation (random rotation and flipping of patterns) is disabled. Enabling it would significantly increase task difficulty and require a much larger training budget to achieve the same level of learning.

**6. `perfect_bonus = False`**
No additional reward for achieving the maximum possible score on a puzzle. Keeping the reward signal purely proportional simplifies the credit assignment problem and makes the reward landscape consistent across levels.

**7. Reward structure: `(index + 1)²`**
Reward is quadratic in how far along the growth pattern the target has expanded at submission time. This creates a strong gradient toward submitting at the optimal moment rather than submitting arbitrarily early.

**8. Observation: flattened grid `(nv × 2nv = 72)`**
The raw game grid is flattened into a 1D vector. No handcrafted feature engineering — the network must learn its own representation from the spatial structure of the board.

**9. Action space: `Discrete(nv + 1) = 7`**
The agent can rotate the hexagon by 0–5 vertices or submit (action 6). Submit is a discrete action the agent must learn to select at the right time, rather than being triggered automatically.

---

## Agent Architecture

**10. DQN and DRQN — not policy gradient methods**
Value-based methods are well-suited to discrete action spaces and sparse rewards. Policy gradient methods (PPO, A2C) tend to require more samples to converge in environments with infrequent reward signals. DQN and DRQN also allow direct comparison of a memoryless vs memory-augmented agent on the same task.

**11. DQN: 2-layer MLP**
Two hidden layers are sufficient to approximate the Q-function for a 72-dimensional input with 7 actions. A single layer risks underfitting the non-linear structure of the board; deeper networks increase training time without clear benefit at this input scale.

**12. DRQN: encoder → LSTM → decoder**
The game has temporal structure — the target pattern grows across steps within an episode. A recurrent network can maintain a memory of prior states, which is useful when the optimal action depends on how the board has evolved, not just its current state.

**13. ReLU activation**
ReLU is the standard default for deep RL networks. It avoids the vanishing gradient problem of tanh/sigmoid and is computationally cheap. No evidence of dying ReLU at this network scale.

**14. Same `hidden_dim` for encoder and LSTM in DRQN**
Simplifies the architecture and the hyperparameter search to a single dimension. Decoupling them would add a dimension to the search space without strong a priori reason to expect asymmetric sizes would help.

**15. `hidden_dim` search space 64–512**
64 is the minimum that can meaningfully represent a 72-dimensional input through compression. 512 is the practical upper bound given training time constraints — larger networks train slower per trial, which compounds across 25 Optuna trials.

---

## Exploration

**16. Epsilon-greedy exploration**
Simple, well-understood, and effective for discrete action spaces. Boltzmann exploration requires calibrating temperature; noisy nets add architectural complexity. Epsilon-greedy provides sufficient exploration at this scale without additional tuning.

**17. `epsilon_start = 1.0`**
The agent begins with fully random actions. This ensures the replay buffer is populated with diverse transitions before any learning occurs, which stabilizes early training.

**18. `epsilon_end = 0.05`**
A floor of 5% keeps a small amount of residual exploration throughout training. Setting it to 0 risks the agent getting locked into a suboptimal deterministic policy with no chance of discovering better actions.

**19. `epsilon_decay = 0.99994`**
Applied per step. At this rate epsilon reaches 0.05 at approximately step 50,000 — 67% of the training budget. This gives the agent meaningful exploration across the majority of training before committing to exploitation. The decay rate is explicitly coupled to `total_timesteps`; changing one requires recalibrating the other.

**20. Decay is per-step not per-episode**
Per-step decay is consistent regardless of episode length. Per-episode decay would cause faster decay in environments with short episodes (early levels) and slower decay in longer ones, creating inconsistent exploration schedules across the curriculum.

---

## Training

**21. `total_timesteps = 75,000`**
The training budget is sized so that well-tuned hyperparameters can succeed (clear multiple levels) but poorly-tuned ones cannot. Too small (e.g. 10,000) and even optimal hyperparameters fail, making the Optuna objective flat. Too large (e.g. 500,000) and even mediocre hyperparameters eventually converge, measuring final performance rather than learning speed.

**22. Adam optimizer**
Adam's adaptive learning rates make it robust across a wide range of learning rate values, which is important in hyperparameter search where `lr` varies across orders of magnitude. SGD requires more careful tuning; RMSProp is similar to Adam but lacks momentum.

**23. MSE loss**
Mean squared error is the standard loss for DQN. Huber loss is more robust to large Bellman errors but requires an additional threshold hyperparameter. MSE is sufficient here given gradient clipping already bounds the update magnitude.

**24. Standard DQN Bellman target**
The vanilla Bellman target `r + γ * max Q_target(s', a)` is used rather than Double DQN. Double DQN reduces Q-value overestimation bias by decoupling action selection from evaluation. This is a known limitation — Double DQN would likely improve stability but was not implemented to keep the architecture comparison clean.

**25. Hard target network update**
Target network weights are copied from the policy network every `target_update` steps. Soft updates (`τ`-interpolation) are smoother but introduce an additional hyperparameter. Hard updates are simpler and standard for DQN.

**26. `target_update` search space 10–500**
Too frequent (< 10 steps) causes the target to move faster than the policy can chase it, destabilizing learning. Too infrequent (> 500 steps) slows credit propagation. The range covers both regimes and lets Optuna find the right frequency for this environment.

**27. Gradient clipping `max_norm = 10`**
Early in training, large Bellman errors produce large loss values that can cause unbounded gradient updates and destabilize learning — particularly in DRQN where LSTM gradients can explode through time. Clipping to norm 10 bounds the update size without suppressing meaningful gradient signal. Applied to both DQN and DRQN for consistency.

**28. `rolling_window = 10`**
Episode rewards are smoothed over a 10-episode window for tracking purposes. 10 balances noise reduction with responsiveness — a wider window would obscure rapid improvements, a narrower one would be too noisy to interpret.

---

## Replay Buffer

**29. Standard uniform replay buffer for DQN — pre-allocated numpy arrays**
The buffer uses pre-allocated numpy arrays `(capacity, obs_dim)` and a circular buffer with a position pointer. Sampling is O(batch_size) via `np.random.randint` and direct array indexing. The previous deque-based implementation was O(capacity) per sample because `random.sample()` on a deque requires converting the full deque to a list first — at 25,000 capacity and training every step this was the primary wall-clock bottleneck. Interface and tensor shapes are unchanged; the only behavioral difference is sampling with replacement instead of without, which is negligible at our buffer sizes.

**29a. Standard uniform replay buffer for DQN — algorithm**
Random uniform sampling breaks temporal correlations between consecutive transitions. Prioritized experience replay (PER) would improve sample efficiency by replaying high-error transitions more frequently, but requires additional hyperparameters and implementation complexity.

**30. `SequenceReplayBuffer` for DRQN**
DRQN requires training on sequences rather than individual transitions so the LSTM can build up hidden state during the forward pass. Episodes are stored in full and fixed-length sequences are sampled from them.

**31. `buffer_size` search space 5,000–25,000**
The lower bound (5,000) ensures the buffer holds enough diverse transitions to break temporal correlations. The upper bound (25,000) is set to one-third of the training budget — any larger and the buffer can never be filled, making those values indistinguishable to Optuna.

**32. DRQN: zero-start hidden state during training**
Hidden state is initialized to zeros at the start of each sampled training sequence. The alternative — storing and replaying hidden states alongside transitions — is more accurate but significantly more complex. Zero-start is the standard practical approach.

**33. DRQN: loss on last step of sequence only**
The sequence is passed through the LSTM to build up hidden state (burn-in), but the Bellman loss is only computed on the final timestep. This avoids computing loss on steps where the hidden state is still uninformative, while still propagating gradients through the full sequence via BPTT.

**34. Hidden state resets every episode**
The LSTM hidden state is reset at the start of each episode but persists across all steps within it. This reflects the natural episode boundary in the game — each puzzle is an independent problem — while allowing the agent to integrate information across steps within a puzzle.

---

## DRQN-Specific

**35. `seq_len` search space 4–12**
Capped at 12 because early-level episodes are typically 6–15 steps long. If `seq_len` exceeds the episode length, `SequenceReplayBuffer` discards those episodes (there's no valid start index), meaning no training data is collected until later levels with longer episodes.

**36. Hidden state updated during random actions**
Even when the agent takes a random action under epsilon-greedy, the LSTM still processes the current state and updates its hidden state. This keeps the hidden state trajectory consistent with what it would be during greedy execution, preventing a systematic divergence between exploration and exploitation behavior.

---

## Hyperparameter Search

**37. Optuna over grid or random search**
Optuna's TPE sampler builds a probabilistic model of the objective landscape and proposes trials that are likely to score well. Grid search scales exponentially with the number of hyperparameters. Random search is unguided. TPE finds good configurations with fewer trials.

**38. TPE sampler**
Tree-structured Parzen Estimator models good and bad regions of the search space separately and proposes points that maximize the ratio of good to bad probability. CMA-ES is better suited to continuous unimodal landscapes; TPE handles mixed discrete/continuous spaces like ours more naturally.

**39. `seed = 42` for TPE**
Fixed seed ensures the random startup trials are the same across runs. This makes the study reproducible — given the same completed trial history, the same future trials will be proposed.

**40. `N_TRIALS = 25`**
TPE uses the first 10 trials for random exploration before its surrogate model activates. This leaves 15 TPE-guided trials. For a 6–7 dimensional search space this is sufficient to identify a strong region — the theoretical ideal is ~10× dimensions (60–70 trials) but TPE has strong diminishing returns past ~20 trials at this scale. Reduced for practical runtime reasons.

**41. `direction = "maximize"`**
The objective is a learning speed score — higher values mean levels were cleared earlier. Maximization is the correct direction.

**42. `load_if_exists = True`**
The study persists across crashes and session restarts. Completed trials are never lost. Re-running the same command resumes the study from where it left off, and TPE rebuilds its model from all prior trials.

**43. `lr` on log scale**
Learning rate varies over orders of magnitude in practice (1e-4 to 1e-2). Sampling on a log scale ensures the sampler explores each order of magnitude proportionally rather than concentrating most trials near the high end of the range.

**44. `lr` range 1e-4 to 1e-2**
Standard range for Adam in deep RL. Values below 1e-4 learn too slowly within the training budget; values above 1e-2 cause instability in early training when Q-values are poorly initialized.

**45. `gamma` range 0.90–0.999**
The game has sparse, delayed rewards — the agent only scores when it submits correctly. Low gamma (e.g. 0.90) heavily discounts future rewards, making it hard for the agent to learn that its earlier rotation actions lead to a later submission reward. High gamma (0.999) maintains credit across the full episode length.

---

## Objective Function

**46. `learning_speed_score` as the Optuna objective**
The goal is agents that learn fast, not agents that eventually converge given unlimited training. `learning_speed_score` directly measures when competency is achieved during the training run, not the final reward after training ends.

**47. Formula: `Σ(total_steps − step_when_level_cleared)`**
Each cleared level contributes `(total_steps − step_cleared)` to the score. Clearing level 1 at step 5,000 with a 75,000-step budget contributes 70,000 to the score; clearing it at step 60,000 contributes only 15,000. Earlier clears are worth more, directly penalizing slow learning.

**48. Returns 0 if no level is ever cleared**
No partial credit is given for near-misses. This is a deliberate design choice: the objective should be 0 for any agent that did not demonstrate actual task mastery. The consequence is that Optuna cannot distinguish between two agents that both failed — addressed by ensuring the training budget is large enough that well-tuned agents do succeed.

**49. Early stopping when all levels cleared**
Training halts as soon as all 6 levels appear in `level_clear_steps`. This prevents the environment's built-in level-1 rollback (which triggers after completing all levels) from contaminating the reward and level metrics with a second pass through the curriculum.

**50. Level-clear threshold: `wave_reward >= max_reward - 15 × len(patterns)`**
The tracker records a level as cleared using the same tolerance the environment uses to advance the agent to the next level. Misaligning these thresholds would cause the agent to complete all levels without the tracker ever recording them, breaking the early-stop condition and corrupting metrics.

---

## Experiment Tracking

**51. MLflow for experiment logging**
MLflow provides structured per-run logging with a UI for comparing trials. It integrates cleanly with Optuna — each Optuna trial maps to one MLflow run. The alternative (Weights & Biases) requires an account and internet access; TensorBoard lacks first-class hyperparameter comparison views.

**52. Per-episode logging: reward, epsilon, level, loss**
These four metrics capture the full learning trajectory. Reward shows task performance; epsilon shows where the agent is in the exploration schedule; level shows curriculum progress; loss shows optimization stability. Loss is `None` until the replay buffer is ready to sample, which is reflected in the logs.

**53. `mlruns/` and `optuna_studies/` gitignored**
Experiment results are machine-local and can be large. Committing them would bloat the repository and create merge conflicts when running on multiple machines. The trade-off is that results must be transferred manually between machines rather than pulled via git.
