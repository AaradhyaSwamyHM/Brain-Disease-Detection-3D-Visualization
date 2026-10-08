"""Train the enhanced BraTS 3-D U-Net and save the best portable .keras model."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from data import BraTSSequence, case_ids
from losses import dice_crossentropy_loss, mean_foreground_dice
from unet3d import simple_unet_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-filters", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-cases", type=int, default=100,
                        help="Train on only the first N cases in manifest order (default: 100; use 0 for all).")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_ids, val_ids = case_ids(args.data, "train"), case_ids(args.data, "val")
    available_train_cases = len(train_ids)
    if args.max_train_cases < 0:
        parser.error("--max-train-cases must be 0 or a positive integer")
    if args.max_train_cases:
        train_ids = train_ids[:args.max_train_cases]
    print(f"Training with {len(train_ids)}/{available_train_cases} training cases and {len(val_ids)} validation cases.")
    train_gen = BraTSSequence(args.data, train_ids, args.batch_size, shuffle=True)
    val_gen = BraTSSequence(args.data, val_ids, args.batch_size, shuffle=False)
    sample_x, _ = train_gen[0]
    if args.resume:
        model = tf.keras.models.load_model(args.resume)
        print(f"Resumed {args.resume}")
    else:
        model = simple_unet_model(tuple(sample_x.shape[1:]), base_filters=args.base_filters)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), loss=dice_crossentropy_loss,
                  metrics=[mean_foreground_dice, "sparse_categorical_accuracy"])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(args.output / "best_model.keras"), monitor="val_mean_foreground_dice", mode="max", save_best_only=True),
        tf.keras.callbacks.ModelCheckpoint(str(args.output / "last_model.keras"), save_best_only=False),
        tf.keras.callbacks.EarlyStopping(monitor="val_mean_foreground_dice", mode="max", patience=20, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(args.output / "history.csv")),
    ]
    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)
    model.save(str(args.output / "final_model.keras"))
    frame = pd.DataFrame(history.history); frame.to_csv(args.output / "history_last_run.csv", index=False)
    plt.figure(figsize=(8, 4)); plt.plot(frame["loss"], label="train loss"); plt.plot(frame["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.savefig(args.output / "loss.png", dpi=160); plt.close()
    print(f"Training complete. Best model: {args.output / 'best_model.keras'}")


if __name__ == "__main__":
    main()
