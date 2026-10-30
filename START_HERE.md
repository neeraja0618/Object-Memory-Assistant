# Hybrid Recognition Edition — update first

1. Close the app.
2. Extract the ZIP elsewhere. Copy all Python files (*.py), batch files (*.bat), requirements.txt and the documentation into your existing project folder, choosing Replace.
3. KEEP your existing config.json, data, .venv and downloaded .pt files. Do not replace config.json when updating. No reference photos or memories are bundled in this ZIP.
4. In the existing project terminal run:

```powershell
.\.venv\Scripts\python.exe install_runtime.py
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_windows.bat
```

The runtime script skips reinstalling if the matching CPU pair already works. It refuses to alter a Python environment outside this project's .venv. A first installation can use setup_windows.bat with Python 3.11.

## Test your two bottles

Your existing photos are reused. Personal matching stays enabled in the demo interface. Use yolo11s.pt and input size 640 for the initial recognition test. Keep unfamiliar-object search off. Use a fixed camera; if testing a hand-held clip, your existing camera_motion_warning=false setting can be retained, with the understanding that old named image zones may be unreliable.

Run:

```powershell
.\diagnose_video.bat
```

Choose your original bottle MP4. Wait until it prints Report saved. The diagnostic processes up to the first 30 seconds using the real models and your saved reference crops. It writes data/diagnostics/<timestamp>/recognition_report.json. Send that report to inspect actual scores, view counts, ambiguity, camera warnings and automatic confirmations. A plain model download/loading failure is also recorded if caught. Native crashes cannot be caught by Python.

The diagnostic uses a temporary memory database; it does not alter your registrations or saved sightings. Cached model/reference features may be created as normal. It is not a recognition accuracy benchmark unless independent ground truth is supplied.

## What changed internally

- Reference appearance now combines DINOv2 structure with central-body colour evidence (80% embedding, 20% colour when colour is available). Colour cannot confirm an item with weak embedding evidence.
- The same reference view supplies both features. This helps distinguish violet/steel bottles while retaining review for identical-looking objects.
- Reference confusion analysis uses the same combined evidence. A warning about similar references does not veto a query that passes the actual score, margin and minimum-view checks.
- Strong personal identity evidence helps associate nearby detections when two objects exchange positions. Conflicting evidence at the same location still invalidates trust and requests review.
- Review shows score threshold, separation margin and number of reference photos.
- Failed personal initialization stays visible in the webcam status instead of silently looking like successful personal recognition.
- Matching CPU PyTorch/Torchvision versions are installed together; doctor checks the actual NMS operation. SciPy is included.
- History image cleanup is throttled and session/time lookups are indexed.

## Demo

My objects -> add cropped photos. Start camera/Open video. Pause before reviewing. Seen objects -> Confirm identity if requested. Find -> last confirmed sighting or explicitly unconfirmed candidate evidence. History -> events/export. Settings -> locations/camera, optional Advanced.

Reference scores are similarities, not probabilities. Search reports last confirmed sighting, not guaranteed current location. At least three distinct reference views are needed for automatic identity acceptance. Recognition can remain uncertain despite many photos. Continuous video recording is not included. Video playback still follows analysed frames; the webcam preview runs independently.

## What was verified

76 automated tests pass and Python files compile. Added tests cover colour separation of otherwise similar embeddings, ambiguous identical items remaining uncertain, colour not overriding weak structural evidence, limited background sensitivity, insufficient references and identity-preserving association across swapped positions.

A manually cropped steel and violet bottle from the opening frame of your uploaded clip each matched its corresponding reference better by colour (see validation/color_feature_check.json). This checks only colour discrimination on two examples. It does not prove the full detector/matcher works or establish an accuracy percentage.

Real model inference, webcam hardware and Windows GUI layout could not be run in the build environment because those model libraries/weights and a display are unavailable. Use the diagnostic on your laptop before reporting accuracy or claiming reliable automatic recognition.
