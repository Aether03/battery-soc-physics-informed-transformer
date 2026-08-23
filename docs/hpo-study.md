# Three hyperparameter searches, and why none of them won

## The question

Hand-tuning a model with an eight-dimensional hyperparameter space and a five-term loss is slow and unprincipled. The obvious move is to hand it to a search. This project ran three, against a shared search space, and compared them against manual tuning.

None of them won. What follows is the accounting and the mechanism.

## Setup

All three searched the same eight parameters over the same ranges, so a comparison between methods isn't confounded by different spaces.

| # | Parameter | Range | Type |
|---|---|---|---|
| 1 | `initLR` | [1e-4, 5e-3] | log-uniform |
| 2 | `attentionDim` | {32, 64, 128, 192, 256} | categorical |
| 3 | `numFilters` | {16, 32, 48, 64} | categorical |
| 4 | `dropoutRate` | [0.05, 0.35] | uniform |
| 5 | `gradClip` | [0.5, 2.0] | uniform |
| 6 | `warmupEpochs` | {20, 30, 40} | categorical |
| 7 | `weightDecay` | [1e-6, 5e-4] | log-uniform |
| 8 | `lambda_end` | [1e-2, 5e0] | log-uniform |

Held fixed because they define the model or the labels rather than tune them: `lambda_Ah=10`, `w_ECM=0.5`, `w_data=1.0`, `alpha_ocv=0.20`, `dSOC_scale=1e-3`, and the three rest-gate thresholds.

`lambda_Ah` is deliberately outside the space. Under the residual formulation it penalises a correction that is already small by construction, and searching it risks the optimiser driving the correction to zero and collapsing the model into plain Coulomb counting, which would score beautifully on the clean metric and be worthless.

## What each method does differently

| | ASHA | BOHB | BayesOpt + PCGrad |
|---|---|---|---|
| Search driver | Optuna TPE | Optuna TPE | MATLAB `bayesopt`, GP surrogate |
| Resource allocation | Successive halving, rungs at 25/75/225/400 epochs | Hyperband, several halving brackets at different aggressiveness | None; every trial runs to completion |
| Acquisition | n/a | n/a | Expected improvement, exploration ratio 0.5 |
| Per-trial training | Fixed-weight loss sum, chunked and resumable | Identical to ASHA | PCGrad gradient surgery, three task gradients per step |
| Trials | 40 | 55 + 12 refinement | 55 |

ASHA and BOHB share a trainer that is mechanically identical, so the two differ *only* in pruning strategy. That was intentional: it makes them a controlled comparison rather than two implementations that drifted apart.

The Bayesian-optimisation arm differs in both search and training. PCGrad computes three separate gradients (SOC fidelity, correction magnitude, ECM consistency) and for each pair with a negative inner product, projects the conflicting component out before summing:

```
if ⟨gᵢ, gⱼ⟩ < 0:   gᵢ ← gᵢ − (⟨gᵢ, gⱼ⟩ / ‖gⱼ‖²) · gⱼ
```

Roughly 3× the backward-pass cost per step. Pairing it with Bayesian optimisation rather than a multi-fidelity scheme is deliberate: expensive trials suit a search that needs few high-signal evaluations, not one built on many cheap partial ones.

## Results

| Method | Trials | Pruned | Best trial RMSE | Compute |
|---|---:|---:|---:|---:|
| ASHA | 40 | 38 | 20.28% | 24.2 h |
| BOHB | 55 | 45 | 13.50% | 48.3 h |
| BayesOpt + PCGrad | 55 | 0 | **2.25%** | 27.9 h |

Best configuration found, by the Bayesian-optimisation arm at trial 49:

```
initLR 4.904e-3   attentionDim 128   numFilters 64   dropoutRate 0.328
gradClip 0.732    warmupEpochs 30    weightDecay 8.8e-5   lambda_end 0.0323
```

## Making the comparison fair

Those trial numbers cannot be read against the baseline's 0.4129% directly, and doing so would flatter the baseline enormously. The harnesses run `seqLen=240` rather than 480, implement the four-term loss without the drift term, and early-stop on the windowed validation metric. Halving the sequence length roughly doubles trial throughput. That is a legitimate search-time trade-off, but it means the search never evaluated the accepted configuration.

The honest comparison is to take the best configuration any search found and run it through the full production pipeline:

| Configuration | Continuous test RMSE |
|---|---:|
| Hand-tuned baseline, V8.6 | **0.4129%** |
| Best searched configuration, re-run at full fidelity | 0.6518% |

Manual tuning wins by 0.24 pp, after roughly 100 hours of search compute.

## Why the searches lost

**The residual error wasn't a hyperparameter problem.** By the time the searches ran, the model's remaining error was accumulated integration drift: a slow bias between anchor points, tiny per step, compounding over thousands of steps. No point in that eight-dimensional space fixes it, because the fix was structural:

1. A drift loss that squares the *sum* of unanchored deviations rather than the mean of squares, which is what makes a 2.6e-5-per-step bias register at the same order of magnitude as the data loss.
2. A model-selection metric that measures continuous reconstruction rather than per-window accuracy, so the selected checkpoint is the one that actually holds over a long horizon.
3. A learning-rate annealing horizon decoupled from the epoch budget, so the schedule reaches its floor before patience-based stopping fires.

Those three changes took validation RMSE from 3.05% to 0.37%. The best of 150 searched configurations took it to 2.25%.

**The searches optimised a metric blind to the problem.** None of the harnesses carry the drift loss or the continuous selection metric. They were searching for a good configuration under a measure structurally incapable of seeing the error that mattered. V8.4 illustrates the failure exactly: 0.39% windowed, 3.05% continuous. A search rewarded on the windowed number would happily converge on a model with metres of drift.

**ASHA specifically got a bad deal from its own protocol.** Its best trial scored 10.55% at epoch 25 and 20.28% at epoch 400. It got worse with more training. The harness returns the value at the milestone epoch rather than the best seen, so within-trial divergence is never rewound, and 38 of 40 trials were pruned on early-rung evidence. Successive halving assumes early performance predicts late performance, and on this objective it does not.

## What this is worth

A negative result with a mechanism is more useful than a positive one without. The finding here is not "automated HPO doesn't work". It is that automated search cannot find a fix that lies outside the space it is searching, and that a search rewarded on a metric blind to your dominant error term will confidently return a bad answer.

The natural next experiment is to re-align the harnesses to the production configuration: add the drift loss, switch selection to the continuous metric, restore `seqLen=480`, and put `lambda_drift` in the search space where `lambda_end` currently sits. Until that runs, the honest claim is narrower than "manual beats automated": it is that these three searches, configured this way, lost to manual tuning, and the reason they lost is legible.

Trial-level data for all three searches is in [`../results/hpo/`](../results/hpo/).
