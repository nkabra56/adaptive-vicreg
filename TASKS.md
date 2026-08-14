# Tasks

Punch list compiled during a repo review pass on 2026-08-12. Update this file
in place as items are resolved.

## Needs your input / external action

- [x] `citation.cff` `repository-code`/`license-url` pointed at placeholder
      URLs. Resolved: filled in from `git remote` (`nkabra56/adaptive-vicreg-cs584-final-proj`).
- [ ] `citation.cff` `orcid` is a placeholder (`0000-0000-0000-0000`). Left as
      a visible placeholder per your choice; register a real ORCID and update
      it later if you want one in the citation record.
- [x] `.gitignore` references `ablations.md`, `slides.md`, and NGC container
      scripts that aren't tracked in git. Resolved: none of these exist on
      disk, so there's nothing to recover or worry about losing.

## Verification needed (run on your training machine, not doable in this sandbox)

- [ ] `pytest tests/` has never actually executed. This dev sandbox's Python
      (3.14) has no installable TensorFlow, so the test suite (losses,
      schedules, `AdaptiveReweighter`) is only syntax-checked and
      cross-validated in numpy so far.
- [ ] Smoke test the new `AdaptiveReweighter` against real training
      (`--epochs 1 --adaptive`) before trusting it on a full run.
- [ ] README's "Experimental results" (section 6) are explicitly marked
      stale, from the old, broken adaptive-targets-only mechanism. Needs a
      fresh baseline vs. `--adaptive` comparison run.

## Known scope gaps (not bugs, just unfinished scope)

- [ ] `--bn-freeze-steps` in `resume_pretrain.py` is accepted but is a no-op.
      `VICRegTrainer` has no BN-freeze mechanism implemented. Either build it
      or drop the flag.
- [ ] Dataset support is CIFAR-10/100 only (hardcoded 50k sample count in
      `steps_for_dataset`).
- [ ] Augmentation set is minimal relative to modern SSL recipes (no
      random-resized-crop, Gaussian blur, or solarization).
- [ ] Architecture is a single fixed CNN encoder. No ResNet-18/50 option, no
      LARS optimizer (the original VICReg paper uses LARS for large-batch
      training).
- [ ] `ModelCheckpoint` monitors train `loss`, not a held-out validation
      metric.
- [ ] No CI configured. Tests exist but nothing runs them automatically on
      push.
