# Train a detector for your objects

Custom training is the route to investigate when stronger pretrained models still repeatedly confuse your actual items. This is a pipeline, not a supplied trained model.

1. Choose a narrow category set: cell phone, book, bottle, cup and scissors.
2. Record several different examples of each, with varied backgrounds, light, scale, orientation and covering. Collect the confusing phone/book and scissors/laptop cases deliberately, plus ordinary examples and images without target objects.
3. Extract well-spaced frames. Do not rely on many almost identical frames from one short clip.
4. Label images in an object-detection annotation tool and export YOLO-format labels. Annotate every visible target item; do not label a genuine laptop as scissors to compensate for a model mistake.
5. Split **whole recording sessions** into train/validation/test so neighbouring frames and identical setups do not leak between sets. Reserve test footage before tuning.
6. Organize matching image and label filenames:

```text
desk_objects/
  images/train/   labels/train/
  images/val/     labels/val/
  images/test/    labels/test/
```

Each label line is `class_id x_center y_center width height`, with normalized coordinates. Empty files can represent images with no target instances. Use your annotation export rather than guessing coordinates.

7. Copy dataset.example.yaml to dataset.yaml and set its path to your actual dataset. Preserve exact class-name spelling.
8. Run:

```powershell
.\.venv\Scripts\python.exe train_custom.py --data dataset.yaml --epochs 50 --device cpu
```

Fifty epochs is a starting configuration, not an accuracy guarantee. CPU training can be slow. For a compatible CUDA training environment, use `--device 0`. That requires a CUDA-capable GPU and compatible PyTorch installation; the app's normal inference is CPU-only.

9. Review validation results and failure cases. Adjust data collection or settings without using the test set for repeated tuning.
10. Evaluate the final best.pt model once on held-out test data using the official detector validation API, and report exact settings and precision/recall/mAP. For memory behaviour, separately run the application on staged held-out videos and measure wrong memories, missed confirmations, last seen location/time errors and false absences.
11. In config.json change remember_classes to the exact names in your model (the example uses five). Restart the app and use Custom weights to select training_runs/.../weights/best.pt.

Corrections in the UI do not automatically train the model. A nickname such as “my black phone” is memory metadata, not a new detector category. The three uploaded application screenshots are insufficient for custom training and should not be mixed into training as ordinary camera frames.
