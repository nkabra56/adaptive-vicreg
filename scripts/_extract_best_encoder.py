"""Re-save encoder weights from a run's best full-trainer checkpoint.

`train_vicreg.py` now reloads the best checkpoint before saving `vicreg_encoder.weights.h5`. Runs made
before that fix saved the encoder from the final epoch instead, even when training had gotten worse since
its best epoch. This loads `vicreg_full.weights.h5` (kept by ModelCheckpoint at the best loss) into a fresh
trainer and writes its encoder to `vicreg_encoder_bestckpt.weights.h5` in the same run directory.

Usage: python3 scripts/_extract_best_encoder.py <run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vicreg_tf import (
    VICRegTrainer,
    VICRegWeights,
    build_encoder,
    build_projector,
    force_build_for_saving,
    safe_load_trainer_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-save encoder weights from a run's best full-trainer checkpoint.")
    parser.add_argument("run_dir", help="Run directory with train_config.json and vicreg_full.weights.h5.")
    run_dir = parser.parse_args().run_dir

    with open(os.path.join(run_dir, "train_config.json")) as f:
        cfg = json.load(f)

    encoder = build_encoder(cfg["image_size"], feat_dim=cfg["feat_dim"])
    projector = build_projector(cfg["feat_dim"], cfg["proj_out"], cfg["proj_layers"])
    # Only the weight layout matters here, and it doesn't depend on the adaptive flags.
    trainer = VICRegTrainer(
        encoder=encoder,
        projector=projector,
        w0=VICRegWeights(sim=25.0, var=25.0, cov=1.0),
        use_schedules=cfg["use_schedules"],
        steps_per_epoch=1,
        epochs=cfg["epochs"],
        base_lr=cfg["lr"],
        base_wd=cfg["wd"],
    )
    force_build_for_saving(trainer, encoder, projector, cfg["image_size"])

    full_ckpt = os.path.join(run_dir, "vicreg_full.weights.h5")
    safe_load_trainer_weights(trainer, full_ckpt)

    out_path = os.path.join(run_dir, "vicreg_encoder_bestckpt.weights.h5")
    encoder.save_weights(out_path)
    print(f"[extract] Reloaded best checkpoint from {full_ckpt}")
    print(f"[extract] Wrote encoder weights to {out_path}")


if __name__ == "__main__":
    main()
