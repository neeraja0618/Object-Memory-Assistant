"""Short-lived camera overlays.

Display tracking does not update confirmed object memory.
"""

import time

import cv2
import numpy as np


class PreviewOverlay:
    def __init__(self):
        self.gray = None
        self.items = []
        self.expires = 0
        self.last_frame = None

    @staticmethod
    def small_gray(frame):
        resized = cv2.resize(frame, (320, 240))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def seed(self, frame, entries, age=0):
        self.gray = self.small_gray(frame)
        self.last_frame = frame
        self.items = []

        # Old detections must not remain on screen indefinitely.
        self.expires = time.monotonic() + max(0, 4 - age)

        for box, label, color in entries:
            box = np.clip(
                np.asarray(box, dtype=float), 0, 1
            )

            x1, y1, x2, y2 = (
                box * [320, 240, 320, 240]
            ).astype(int)

            mask = np.zeros((240, 320), np.uint8)
            mask[y1:y2, x1:x2] = 255

            points = cv2.goodFeaturesToTrack(
                self.gray,
                maxCorners=30,
                qualityLevel=0.02,
                minDistance=5,
                mask=mask,
            )

            self.items.append(
                [box, label, color, points]
            )

    def render(self, frame):
        if (
            self.gray is None
            or time.monotonic() > self.expires
        ):
            return frame

        if frame is not self.last_frame:
            gray = self.small_gray(frame)
            kept = []

            for box, label, color, points in self.items:
                if points is None or len(points) < 4:
                    continue

                moved, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.gray,
                    gray,
                    points,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                )

                if moved is None:
                    continue

                back, reverse, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    self.gray,
                    moved,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                )

                if back is None:
                    continue

                good = (
                    (status.ravel() == 1)
                    & (reverse.ravel() == 1)
                    & (
                        np.linalg.norm(
                            back - points, axis=2
                        ).ravel()
                        < 1.5
                    )
                )

                if good.sum() < 4:
                    continue

                delta = np.median(
                    (moved - points)[good, 0],
                    axis=0,
                ) / [320, 240]

                if np.linalg.norm(delta) > 0.25:
                    continue

                box = np.clip(
                    box
                    + [
                        delta[0],
                        delta[1],
                        delta[0],
                        delta[1],
                    ],
                    0,
                    1,
                )

                if (
                    box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    continue

                kept.append(
                    [box, label, color, moved[good]]
                )

            self.items = kept
            self.gray = gray
            self.last_frame = frame

        output = frame.copy()
        height, width = output.shape[:2]

        for box, label, color, _ in self.items:
            x1, y1, x2, y2 = (
                box * [width, height, width, height]
            ).astype(int)

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                output,
                label + " [tracked]",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return output