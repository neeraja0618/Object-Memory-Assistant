# Object Memory Assistant — Hybrid Recognition Edition

See START_HERE.md for safe update instructions and the diagnostic command.

The application detects supported objects, matches registered physical items using reference appearance, associates observations into temporary tracks and stores confirmed last-seen evidence. Uncertain identities remain separate from confirmed memories and can be reviewed. Personal registration creates a reference gallery; it does not retrain YOLO.

Core modules: base_app.py (source/GUI), personal.py and appearance.py (reference matching), engine.py/personal_engine.py/research_engine.py (tracking and memory), research.py (reference guidance), app.py (demo UI and history), diagnose_video.py (real-model diagnostics).

The shipped tests validate software behavior, not end-to-end recognition accuracy. Read START_HERE.md for known limitations, and RESEARCH_PLAN.md for evaluating the project's claims. Existing public dataset/training/evaluation utilities remain available.
