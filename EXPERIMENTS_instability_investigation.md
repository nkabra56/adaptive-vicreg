# Instability investigation (2026-10-01)

Mining the 13 existing per-epoch `metrics/history.jsonl` files for a leading indicator of the
loss/`avg_std` spike described in EXPERIMENTS.md findings 1, 4, 10, 12, 13, 14, 17, 18. No new
training runs were done for this pass, everything below comes from data already on disk. The
analysis script used to produce every number in this file is not checked in (it was a disposable
one-off in a scratch directory), but every table below was generated directly from the
`history.jsonl` files named.

## Step 1: relabeling the runs, and one correction to EXPERIMENTS.md

Loaded all of `final-baseline-s0`, `final-baseline-s1` (+ `final-baseline-s1_resumed`, concatenated
by epoch, gap at epoch 62 as expected), `baseline-s2/s3/s4`, `batch128-s0/s1`,
`vareps-s3recipe/s4recipe`, plus `adaptive-fixed-s0/s1` and `targets-s0/s1` as supplementary context
(different objective, not part of the main spike/clean comparison). A spike was flagged
programmatically as: the first epoch (after an epoch-6 cutoff, to skip the universal collapse-recovery
ramp every run does in its first few epochs) where loss or `avg_std` jumps more than 3x or 10x
respectively over the trailing 3-epoch median, AND the metric falls back by at least half within the
next 3 epochs (the actual shape of every known spike: a sharp jump, then partial or full recovery,
as opposed to a plain monotonic drift).

Result, compared against EXPERIMENTS.md's findings 12/13/17/18:

| Run | My label | My spike epoch | EXPERIMENTS.md label | EXPERIMENTS.md epoch |
|---|---|---|---|---|
| baseline-s0 | clean | - | clean | - |
| baseline-s1 | spiked | 14 | spiked | 14 |
| baseline-s2 | clean | - | clean | - |
| baseline-s3 | spiked | 14 | spiked | 14 |
| baseline-s4 | spiked | 17 | spiked | 17 |
| batch128-s0 | clean | - | clean | - |
| batch128-s1 | **spiked** | **11** | "gradual degrade, no spike" | - |
| vareps-s3recipe | spiked | 13 (14 by the loss criterion alone) | spiked | 14 |
| vareps-s4recipe | spiked | 14 | spiked | 17 |

This agrees with EXPERIMENTS.md everywhere except **batch128-s1**, and I think the write-up undersold
what's in that file. Its `loss/total` goes 118.29 (epoch 10) -> **700.47** (epoch 11, a 5.9x jump) ->
233.65 -> 186.11 -> ... -> 124.4 (epoch 30), and `loss/align` does 0.730 -> **4.081** (5.6x) in the
same step, with `loss/cov` jumping 7x (0.0207 -> 0.1445). That is the same shape as every other
spike in this dataset: a sharp one-epoch jump in `loss/total` and `loss/align`, immediately followed
by partial recovery. What's different is that `stats/avg_std` does **not** spike upward with it
(0.810 -> 0.732, actually a small dip), because this run's `avg_std` was already on its own separate
gradual decline (0.81 at epoch 10 down to ~0.42-0.47 by epoch 25-30, exactly as EXPERIMENTS.md
finding 17 describes). So finding 17's framing ("no single epoch standing out the way loss jumps by
10x+") is accurate for `avg_std` specifically, but there plainly is a single standout epoch in
`loss/total` and `loss/align` (just a 5.9x jump, not 10x+, and with `avg_std` not following it
upward). My read: this run had **both** the sharp align/cov-driven spike mechanism (same as the
batch-256 runs) **and** a separate, slower variance-collapse drift, and they happened to overlap
around epoch 10-11. Vareps-s4recipe's epoch count also disagrees slightly with EXPERIMENTS.md (14
here against 17 there) depending on whether the loss criterion or the `avg_std` criterion is used as
the trigger; both land within the same 2-epoch window of the real event (`avg_std` itself jumps at
epoch 13 in that run, loss follows at 14), so I don't think this is a real disagreement, just two
reasonable ways to pick "the" spike epoch from one fast-moving event.

## Step 2: looking for a leading indicator in the 2-4 epochs before each spike

### Absolute thresholds: none found

Comparing the epoch immediately before each spike against the same loss terms and stats at the
same, or a nearby, epoch number in the two confirmed-clean runs (baseline-s0, baseline-s2):

| Run | epoch before spike | align | var | cov | avg_std | corr_sq |
|---|---|---|---|---|---|---|
| baseline-s1 | 13 | 0.657 | 0.682 | 0.0110 | 0.744 | 0.0096 |
| baseline-s3 | 13 | 0.700 | 0.585 | 0.0117 | 0.883 | 0.0097 |
| baseline-s4 | 16 | 0.769 | 0.444 | 0.0128 | 1.027 | 0.0191 |
| vareps-s3recipe | 12 | 0.642 | 0.716 | 0.0108 | 0.709 | 0.0083 |
| vareps-s4recipe | 13 | 0.888 | 0.250 | 0.0168 | 2.723 | 0.1037 |
| baseline-s0 (clean), epoch 13 | - | 0.687 | 0.605 | 0.0144 | 0.746 | 0.0097 |
| baseline-s2 (clean), epoch 13 | - | 0.564 | 0.878 | 0.0110 | 0.574 | 0.0061 |
| baseline-s0 (clean), epoch 16 | - | 0.741 | 0.401 | 0.0141 | 0.859 | 0.0076 |

Every one of these ranges overlaps completely between the about-to-spike runs and the clean runs at
comparable epochs. `baseline-s1`'s pre-spike `align`/`avg_std` (0.657 / 0.744) is nearly identical to
clean `baseline-s0`'s value at the exact same epoch (0.687 / 0.746). No fixed value or ratio of any
logged quantity, checked one epoch before the event, separates a run that is about to spike from one
that never will. (vareps-s4recipe is the outlier with `avg_std` already at 2.72, but that's because
its chosen "epoch before" is already inside the accelerating ramp described next, not a stable
baseline value.)

### Weighted covariance contribution (`loss/w_cov * loss/cov`): nothing unusual pre-spike

Findings 1/4/6 are about covariance-loss scaling, so this was checked directly even though no prior
finding looked at the *weighted* cov term's trajectory. In every run (spiking or clean), the weighted
cov contribution (`w_cov=4095 * loss/cov`) sits in the same 40-65 range in the epochs before whatever
happens next, and if anything is slightly **lower** in the about-to-spike runs (42-48) than in clean
`baseline-s0` at the same epoch (57-65). It only explodes *during* the spike epoch itself (reaching
570-1028), as a symptom of `loss/cov` jumping, not a precursor. This rules out "the weighted cov term
creeps up beforehand" as a mechanism, at least at epoch resolution.

### Step-count (not epoch-count) clustering: ruled out

The task's concrete hypothesis: do spike epochs cluster when converted to gradient-step count, given
batch128 has ~2x the steps/epoch of batch256 (390 vs 195, from `50_000 // batch_size`)? No.

| Run | batch size | steps/epoch | spike epoch | spike step range |
|---|---|---|---|---|
| baseline-s1 | 256 | 195 | 14 | 2535-2730 |
| baseline-s3 | 256 | 195 | 14 | 2535-2730 |
| baseline-s4 | 256 | 195 | 17 | 3120-3315 |
| vareps-s3recipe | 256 | 195 | 13 | 2340-2535 |
| vareps-s4recipe | 256 | 195 | 14 | 2535-2730 |
| batch128-s1 | 128 | 390 | 11 | 3900-4290 |

The batch256 runs cluster loosely around step 2500-3300. The one batch128 spike lands at step
3900-4290, outside that whole range, not inside it. Two batch-size groups, one data point for the
smaller batch, is not much to generalize from, but it gives no support to the step-count hypothesis
and some evidence against it: if anything, batch128 spikes later in step count than batch256, the
opposite of what you'd expect if the spike were simply the Nth optimizer update regardless of epoch.
Epoch number alone (11, 13, 14, 14, 14, 17) remains the only accounting in which these events cluster
at all, and even that clustering (three 14s) is weak given seed 4's two events land at 17.

### Epoch-over-epoch acceleration: a real but weak and short-lead signal

This is the one place something distinguishes the two groups, but only barely and only very close to
the event. Looking at `loss/align` and `stats/avg_std` over the 4 epochs immediately before each
spike, both quantities are not just rising (which happens in every healthy run too, as variance
recovers from its early-training floor) but **accelerating**: each epoch's delta is 1.5-4x the
previous epoch's delta, for all 6 spike events including batch128-s1's. Concretely, `loss/align`'s
four pre-spike deltas for baseline-s1 are `0.0104, 0.0115, 0.0298, 0.0728` (successive ratios 1.1,
2.6, 2.4); for baseline-s4, `0.0126, 0.0214, 0.0379, 0.1453` (ratios 1.7, 1.8, 3.8).

The problem: clean runs show the *same qualitative* acceleration during the analogous part of
training, just usually smaller in the final step. `baseline-s2` (confirmed clean for its full 100
epochs) over epochs 12-16 has `loss/align` deltas `0.0075, 0.0112, 0.0144, 0.0421`, ratios `1.5, 1.3,
2.9`, which is the same shape (and a similar final ratio) as several of the spiking runs. The
difference is one of magnitude in the very last step before the event (spiking runs' final delta is
roughly 0.06-0.22, clean runs' is roughly 0.01-0.04, a 3-5x gap), not of kind. By the time that gap is
visible, the event is one epoch away at most. This is consistent with (not independent evidence
against) the idea that the acceleration IS the leading edge of the spike itself, already underway
within the "pre-spike" epoch, rather than a true multi-epoch precursor. It is not usable as an early
warning at epoch resolution: there's no clean threshold on the acceleration itself that avoids false
positives against runs like baseline-s2, and even a perfect detector of it would give less than one
epoch of lead time.

### What this rules out, concretely

At per-epoch resolution: no absolute value or ratio of `loss/align`, `loss/var`, `loss/cov`,
`stats/avg_std`, `stats/avg_offdiag_corr_sq`, or the weighted cov contribution, checked 1-4 epochs
before a spike, separates spiking runs from clean ones. The step-count-clustering hypothesis has no
support in this data. Epoch-number clustering is weak (three events at 14, but also 11, 13, 17). The
only thing that correlates with an incoming spike is the rate of change of `loss/align`/`avg_std`
over the last 1-2 epochs accelerating, and clean runs do a smaller version of the same thing, so it
isn't a reliable binary signal, only a (noisy) matter of degree, visible too late to be useful as
built.

## Step 3: is per-epoch resolution fine-grained enough? No.

The actual jump happens somewhere inside a single recorded epoch: `loss/align` goes from order-0.6-0.9
to order-4-14 within one epoch boundary (195-390 batches), and it has fully reversed most of the way
back by the next epoch's end. An epoch-level snapshot only ever catches "before" and "after"; it
cannot see which batch, or cluster of batches, inside that epoch the mechanism actually fires in, and
cannot distinguish "building gradually across the whole epoch" from "triggered by one bad batch
partway through." The acceleration signal in Step 2 is itself evidence that something is already
building within the pre-spike epoch, which is exactly the kind of within-epoch structure per-epoch
logging is blind to. A follow-up run needs per-batch (or at least much-more-frequent-than-per-epoch)
instrumentation to have any chance of seeing the actual trigger rather than its aftermath.

## Step 4: new instrumentation for a follow-up GPU run

Added `PerBatchDiagnosticsLogger` to `src/vicreg_tf/callbacks.py` (new class, `VicRegMetricsLogger`
itself is untouched), exported from `src/vicreg_tf/__init__.py`, and wired into
`scripts/train_vicreg.py` behind an opt-in flag with zero effect when it's off.

It follows `VicRegMetricsLogger`'s existing pattern (forward the fixed probe batch through the
encoder/projector, read off embedding stats) but runs on `on_train_batch_end` instead of
`on_epoch_end`, at a configurable stride, and logs three things per-epoch logging can't:

- `probe/avg_std` and `probe/min_std`: not just the mean per-dimension std (what `VicRegMetricsLogger`
  already logs) but the single smallest per-dimension std in the probe batch. A mean can look fine
  while one or two dimensions are already near collapse; the minimum is the one `variance_loss`'s
  `relu(gamma - std)` actually penalizes hardest, and a dimension already near 0 is exactly the
  non-finite-gradient regime finding 14 found at the loss-function level.
- `probe/grad_norm_last_layer`: the gradient norm of `variance_loss + covariance_loss` (evaluated on
  the probe batch) with respect to the encoder's last layer's own trainable variables (found by
  walking `encoder.layers` backward to the last one with trainable variables, not hardcoded to a
  specific layer name, so it isn't tied to `build_encoder`'s current architecture). A spike that
  originates in a bad gradient should show up here before it shows up in the loss, since the gradient
  is upstream of the weight update that (presumably) causes it.
- `loss/*_running_mean`: the same loss-term keys `VicRegMetricsLogger` logs, read straight from Keras'
  per-batch logs. These are the trainer's own cumulative-mean-within-epoch values (Keras resets the
  underlying `keras.metrics.Mean` trackers each epoch), not instantaneous per-batch values, logged
  here only for convenience alongside the finer probe-based numbers above. Getting a true
  instantaneous per-batch loss on the real training batch (as opposed to the fixed probe batch) would
  need `VICRegTrainer.train_step` itself to expose it, which this change deliberately avoids touching
  to keep the instrumentation a pure, zero-risk addition.

Output goes to `<run_dir>/metrics/history_batches.jsonl`, one JSON record per logged batch, with
`epoch`, `batch` (0-indexed within the epoch), and `global_batch` (cumulative) so it can be joined
against `history.jsonl` by epoch, or plotted on its own step axis.

Gamma, nu and `var_eps` are passed in as constants at construction (matching whatever the run uses,
default `gamma=1.0, nu=0.0`), not read live off `AdaptiveTargets`. For the baseline recipe this is
exact; it would be slightly approximate for a `--adaptive-targets` run, which isn't the recipe this
instrumentation is meant for.

### CLI flag

```
--per-batch-diagnostics              Off by default. Enables the callback above.
--per-batch-diagnostics-every N      Log every N training batches (default 10). Smaller N = finer
                                      resolution but more forward+backward passes on the probe batch,
                                      so slower training.
```

### Command for the actual follow-up run

To go after the epoch-14 spike specifically (seed 1 or seed 3's exact recipe, which spiked at epoch
14 twice already), something like:

```bash
python3 scripts/train_vicreg.py \
  --dataset cifar10 --image-size 32 --epochs 100 --batch-size 256 \
  --lr 0.003 --warmup-epochs 5 --w-cov 4095 --use-schedules \
  --seed 1 --metrics-compute-on projector --metrics-probe-batch 256 --record-every 1 \
  --per-batch-diagnostics --per-batch-diagnostics-every 5 \
  --model-dir checkpoints_tf --run-name diag-s1recipe
```

`--per-batch-diagnostics-every 5` gives 39-78 extra probe passes per epoch (195 steps/epoch / 5) for
the 100-epoch run, cheap enough to run the whole thing, but dense enough (every 5 of ~195 steps) to
localize a trigger to roughly a 5-batch window inside whichever epoch it happens in. For an even
tighter look once the approximate epoch is known from a first pass, `--stop-epoch` could bound the run
to just the epochs around the known event with `--per-batch-diagnostics-every 1`, trading full-run
coverage for per-batch resolution across the whole suspect epoch.

## Summary

The per-epoch data rules out several concrete, checkable hypotheses (step-count clustering, a fixed
absolute threshold on any logged quantity, an anomaly in the weighted covariance term) and relabels
one run (`batch128-s1` did have a sharp, EXPERIMENTS.md-undercounted loss/align spike at epoch 11,
layered on top of its separately-described gradual `avg_std` decline). The one real signal found
(accelerating `loss/align`/`avg_std` growth) is present in every spike and does distinguish spiking
from clean runs in degree, but not in kind, and only becomes visible roughly one epoch before the
event, which is both too late and too unreliable (clean runs show a smaller version of the same
acceleration) to use as a predictor. The root cause is still unknown. It most likely lives inside the
handful of batches immediately preceding and during the jump, which per-epoch logging cannot resolve,
hence the new `PerBatchDiagnosticsLogger` and `--per-batch-diagnostics` flag above, ready for a
follow-up GPU run targeting one of the known epoch-14 recipes.
