# VWA Entropy-Based Attack Detection Summary

## Goal

We want to transfer our previous **entropy-based attack detection** idea from AgentDojo to the **VisualWebArena / VWA-Adv** setting.

We are **not** reusing the paper's ARE analysis framework.
We are also **not** using our previous structural / rule-based features.

The focus is only:

- run agents on **VWA benign tasks** and **VWA-Adv attacked tasks**
- collect **token-level logprobs** during generation
- compute **entropy-based signals** from those logprobs
- test whether entropy can detect **attacked trajectories** vs **benign trajectories**
- optionally study **when** in the trajectory the anomaly appears

---

## Previous Core Idea from AgentDojo

Our previous detection idea was:

1. For each agent run, capture the model's **top-k token logprobs** at generation time.
2. Convert token probabilities into **entropy-based uncertainty signals**.
3. Aggregate entropy at different granularities:
   - token-level
   - step-level
   - sliding-window / local window
   - full-trajectory summary
4. Use these entropy features to distinguish:
   - `noattack / benign`
   - `attack`
5. Evaluate detection performance with metrics such as:
   - ROC-AUC
   - Average Precision
   - balanced repeated subsampling when class counts are very imbalanced

The key intuition was:

> attacks do not always change the final answer only; they often perturb the model's local token distribution during generation, and this can appear as abnormal uncertainty patterns.

---

## Why Entropy Instead of Only Final Success / Failure

A trajectory may already look suspicious **before** the final attack succeeds.
So we care about **detection from the generation process**, not only final task outcome.

Entropy is useful because it can capture:

- local uncertainty spikes
- unstable decision regions
- unusual generation behavior around critical action text
- early signs of compromise before the final action is completed

This is especially useful for agent settings where the model repeatedly produces:

- reasoning text
- action strings
- element references
- typed text
- tool / browser interaction commands

---

## What We Actually Want to Transfer to VWA

We want to keep the **same detection philosophy**, but move it to a more realistic benchmark:

- **benign VWA tasks**
- **VWA-Adv adversarial tasks**

So the method stays conceptually simple:

1. run the VWA agent
2. intercept generation logprobs
3. compute entropy features
4. compare attacked runs vs benign runs

---

## Detection Target

The main detection task is:

- **input:** one full or partial VWA trajectory
- **output:** whether it is attacked or benign

Possible label settings:

### Setting A: attack presence detection
Positive if the run is in an attacked environment, regardless of whether the attack finally succeeds.

### Setting B: attack success detection
Positive only if the adversarial objective is actually achieved.

For the first implementation, **Setting A** is easier and more aligned with uncertainty detection.

---

## Data Pairing Recommendation

For each attacked VWA-Adv task, the clean baseline should be as matched as possible.

Best practice:

- same original task
- same agent
- same initial state if possible
- one run with adversarial trigger
- one run with clean environment

This avoids learning confounders such as:

- task difficulty
- website domain style
- trajectory length differences unrelated to attack

---

## Core Entropy Detection Signals

### 1. Token entropy
For each generated token position `t`, if the model provides top-k probabilities `p_1, ..., p_k`, define:

\[
H_t = - \sum_{i=1}^k p_i \log p_i
\]

If the API only gives top-k rather than full-vocabulary probabilities, this is a **truncated entropy proxy**, which is still useful in practice if used consistently.

### 2. Step-level entropy
Each agent step may contain a generated response or action string.
Aggregate token entropies within one step using:

- mean entropy
- max entropy
- tail mean entropy
- entropy standard deviation

### 3. Sliding-window entropy
This was the main idea from our previous AgentDojo work.
Instead of only averaging over the whole trajectory, compute local entropy windows.

For a token sequence `H_1, H_2, ..., H_T`, define window summary:

\[
W_j = \frac{1}{w} \sum_{t=j}^{j+w-1} H_t
\]

and sweep different window sizes `w`.

Why this matters:

- attacks may only affect a local region
- full-trajectory averaging may dilute the signal
- local windows can expose short abnormal bursts

### 4. Early-stage entropy
Because early warning matters, we should also compute:

- entropy from first `N` tokens
- entropy from first `K` agent steps
- first entropy spike position

This helps answer whether attack can be detected before completion.

---

## Minimal Feature Set for the First VWA Version

Codex should first implement a very small set:

- `token_entropy`
- `step_entropy_mean`
- `step_entropy_max`
- `window_entropy_mean`
- `window_entropy_max`
- `trajectory_entropy_mean`
- `first_spike_step`

This is enough to check feasibility before adding anything else.

---

## Where Logprobs Are Needed in VWA Code

The main engineering question is:

> where in the VWA agent code can we intercept token logprobs for each model generation?

Codex should inspect the following layers in the VWA codebase.

### 1. Model invocation wrapper
Find the function/class that actually calls the language model.
This is the highest-priority location.

Typical places to inspect:

- agent / model wrapper
- LM client abstraction
- OpenAI / local model adapter
- generation utility used by the browser agent

We need to determine:

- does the current backend already expose `logprobs`?
- if not, can the backend be configured to return them?
- where is the raw generation response parsed?

### 2. Action generation step
In VWA-style agents, the model usually generates one response per decision step.
Codex should find the function where the agent produces:

- next action
- browser command
- typed text
- final answer

This is the correct place to attach logging because one model call often corresponds to one semantic agent step.

### 3. Response parsing layer
Even if the final action string is postprocessed later, entropy should be computed from the **raw generation output probabilities**, not from parsed actions only.

So Codex should identify:

- raw model output object
- token list
- top-k logprob list per token
- mapping from one model call to one agent step

---

## What Codex Should Look For in the Code

When reading VWA code, Codex should answer these concrete questions:

1. **Where is the LLM called?**
2. **Which model backend is used?**
3. **Can this backend return per-token logprobs or top-k logprobs?**
4. **Where is one agent decision step generated?**
5. **How can we save, for every step:**
   - generated text
   - token strings
   - top-k logprobs
   - step index
   - trajectory id
   - task id
   - attack/benign label
6. **How can we serialize the results for offline entropy computation?**

---

## Preferred Logging Format

A simple JSON structure per trajectory is enough.

Example:

```json
{
  "task_id": "...",
  "trajectory_id": "...",
  "label": "attack",
  "attack_type": "text_trigger",
  "steps": [
    {
      "step_idx": 0,
      "prompt_summary": "...",
      "generated_text": "...",
      "tokens": [
        {
          "token": "Click",
          "topk": [
            {"token": "Click", "logprob": -0.10},
            {"token": "Type", "logprob": -1.20},
            {"token": "Scroll", "logprob": -2.00}
          ]
        }
      ]
    }
  ]
}
```

Then entropy can be computed offline from these logged top-k probabilities.

---

## Offline Entropy Computation

For each token's top-k logprobs:

1. convert logprobs to probabilities
2. renormalize over available top-k if needed
3. compute entropy
4. aggregate by step / window / trajectory

Pseudo-formula:

\[
p_i = \frac{\exp(\ell_i)}{\sum_j \exp(\ell_j)}
\]

\[
H = -\sum_i p_i \log p_i
\]

If top-k is already normalized by the backend, then renormalization may not be necessary.
Codex should verify backend behavior carefully.

---

## Recommended First Evaluation Plan

### Phase 1: Feasibility
Use a small subset of VWA benign tasks and VWA-Adv attacked tasks.
Check only:

- whether logprobs can be extracted reliably
- whether entropy can be computed for every step
- whether attacked runs show visible differences from benign runs

### Phase 2: Paired detection
For matched benign/attacked runs:

- compute trajectory-level entropy summaries
- compute step-level entropy curves
- compare distributions
- report ROC-AUC / AP

### Phase 3: Prefix detection
Check whether the first few steps are already sufficient for detection.
This is important because early warning is more useful than post-hoc judgment.

---

## Main Engineering Objective for Codex

The immediate goal is **not** to redesign the benchmark.
The immediate goal is:

> inspect VWA code, find the exact generation call path, and determine the cleanest place to capture token logprobs so that entropy-based detection can be implemented with minimal code changes.

So Codex should focus on:

- tracing the model call stack
- confirming logprob availability
- identifying step boundaries
- proposing the smallest logging patch needed

---

## Final One-Sentence Summary

We previously developed an **entropy-based attack detection idea** in AgentDojo by extracting token-level top-k logprobs, converting them into local uncertainty signals, and aggregating them over steps and sliding windows to distinguish attacked from benign runs; now we want to apply the same idea to **VWA / VWA-Adv**, and the main coding question is simply **where in the VWA agent pipeline to intercept generation logprobs so entropy can be computed per token, per step, and per trajectory**.

