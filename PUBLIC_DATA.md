# Use public data without recording your own objects

The official ORBIT-India record is https://zenodo.org/records/12608444 . It contains clean/clutter image frames derived from videos and presence/privacy annotations. The verified download link advertised a 1,295,230,307-byte ZIP (~1.3 GB) with MD5 af4d05689f556ca477acc09a73bb3d3d. The build environment did not download or benchmark this archive. The provided tool runs on your laptop; allow download/extraction space and inspect the source's dataset terms before research use.

## Download and prepare

After setup:

```powershell
.\.venv\Scripts\python.exe public_data.py download --destination datasets
.\.venv\Scripts\python.exe public_data.py manifest datasets\orbit_india --output evaluation\public_manifest.json
```

The downloader verifies the published checksum and extracts with archive-path checks. It is not resumable; a failed partial download restarts. If the source/version changes, check the official record rather than suppressing the checksum error.

The manifest selects one collector with at least two objects having both clean and clutter examples, samples up to six objects, uses clean reference images and separate clutter query images, and uses presence/privacy annotations when available. Options include --collector P1 and --max-objects 3. Missing frame-presence annotations are explicitly reported; video/folder labels do not prove target presence in every frame.

## Appearance-component experiment

```powershell
.\.venv\Scripts\python.exe evaluate_public.py evaluation\public_manifest.json --budget 5 --output evaluation\results
```

This actually extracts features and writes predictions.csv, reference_analysis.json and summary.json on your machine. It compares nearest matching, uniform fixed-threshold matching, guided fixed-threshold matching and guided adaptive-threshold matching, with the same reference budget. Selection/adaptation uses reference data only, not clutter queries.

This is a **custom full-frame appearance diagnostic**, not the official ORBIT benchmark, detector mAP, memory-history evaluation or proof of research novelty. It does not localize the query object first; background can influence features. Clean/clutter examples are separated but the subset protocol must be documented. Report coverage and accepted errors, not just accepted precision.

## Seed the app's public-demo registrations

```powershell
.\.venv\Scripts\python.exe seed_public_registry.py evaluation\public_manifest.json --views 5
```

Objects are prefixed “Public demo:”. Their stable registration IDs are written to evaluation/public_registration_ids.json. Full reference frames may contain background; inspect/crop examples if needed. This demonstrates dataset objects, not your own keys or bottle.

## Make a playable sequence

Choose one object's clutter folder in the extracted dataset. List genuine recording prefixes:

```powershell
.\.venv\Scripts\python.exe make_sequence_video.py "PATH_TO_ONE_OBJECT_CLUTTER_FOLDER"
```

Then choose exactly one listed prefix:

```powershell
.\.venv\Scripts\python.exe make_sequence_video.py "PATH_TO_ONE_OBJECT_CLUTTER_FOLDER" --prefix "EXACT_LISTED_PREFIX" --fps 10 --output public_demo_sequence.mp4
```

Open that MP4 in the app. The tool preserves file/frame order within one recording. The supplied playback FPS is an assumption, not the original timing. Do not join unrelated photos or recordings and call it a real temporal-memory benchmark. ORBIT footage may use a moving camera; the fixed-camera scene-warning heuristic may suspend monitoring. Document any choice to disable it for a recognition-only demo.

## Evaluate memory and repair separately

Export observations from a real session, make a label template, inspect evidence and independently label target identities:

```powershell
.\.venv\Scripts\python.exe evaluate_memory.py observations_and_repairs.json --make-label-template ground_truth.csv
.\.venv\Scripts\python.exe evaluate_memory.py observations_and_repairs.json --labels ground_truth.csv
```

Set labelled=1 only for independently inspected rows. For those rows, blank object_id means no registered target. Leave labelled=0 for unlabelled rows. Use registration IDs, not names. Optional zone labels use the application's nine image zones, not 3D coordinates. The tool reports observation-level acceptance errors/coverage, automatic/manual counts, final memory coverage and time error; optional zone accuracy is only computed when independently supplied. It is not a complete room-search benchmark.

Before/after repair diagnostics reconstruct earliest pre-repair states from the audit. User corrections are supervision: improvement after supplying ground truth must not be reported as an unsupervised model breakthrough. Compare policies on the same underlying footage and keep final test recordings out of tuning.

ORBIT's original source and official protocol: https://github.com/microsoft/ORBIT-Dataset . Cite the dataset creators; do not claim this project collected the public data.
