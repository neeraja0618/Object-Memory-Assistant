# Verification limits and evaluation

The final software tests cover temporal memory, reference storage/matching, identity guards, camera-transform math, simulated source/inference cleanup, confusion guidance, selective and always-review policies, review budgets, tentative history, bounded repair, preserved originals, stale/expired preview rejection, repaired-evidence retention, public manifest splits and evaluation metric calculations. TEST_RESULTS.txt records the executed run.

Tests use synthetic feature vectors, temporary generated evidence images and simulated source/model outputs. They do not measure actual YOLO/DINOv2/OWLv2 accuracy, real camera-shift reliability, webcam hardware, video codecs, Torch/torchvision compatibility or rendered desktop layout. Python syntax checks were also executed. Real end-to-end checks still need the target laptop.

No public dataset or pretrained model was downloaded or evaluated in the build environment. Public-data scripts are provided to run on the laptop; no fabricated benchmark scores are included.

Follow PUBLIC_DATA.md for appearance-component experiments and RESEARCH_PLAN.md for controlled memory evaluation. Use independently labelled evidence, hold models constant in policy comparisons, keep final test sessions separate from reference/tuning data, and report acceptance errors together with coverage and user effort. Separate automatic from manual identity claims and never present user-supplied corrections as unsupervised recognition accuracy.
