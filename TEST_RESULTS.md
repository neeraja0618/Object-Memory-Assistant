# Hybrid Edition validation

76 automated tests passed; compileall passed. New cases test colour/structure combination, ambiguous identical objects, rejection of weak embedding evidence despite colour agreement, central crop background robustness, minimum reference views and nearby crossing association.

validation/color_feature_check.json records a two-crop colour-only check on the opening frame of the user's clip. Reference and observation regions were manually selected. It is neither an automatic detection evaluation nor a measured recognition accuracy result.

Full DINOv2/YOLO inference and Windows webcam/GUI testing could not be performed in this environment. diagnose_video.bat runs the actual pipeline on the user's laptop and writes reproducible scores, configuration and source information for further evaluation.
