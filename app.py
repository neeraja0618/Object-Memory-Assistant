"""Full-scene evidence and consistent identity labels."""

from datetime import datetime
import time
import json
import tkinter as tk
from tkinter import ttk, messagebox

import cv2

from recognition_app import App as RecognitionApp
from base_app import App as BaseApp, ROOT
from demo_search import possible_sightings
from identity_view import (
    valid_box,
    position_description,
    display_identity,
)
from preview_overlay import PreviewOverlay


GENERAL_CLASSES = ["cell phone", "bottle", "book", "cup", "remote", "backpack",
                   "handbag", "scissors", "laptop", "mouse", "keyboard"]


def detector_profile(config, mode=None):
    """One automatic profile using the existing general and specialist weights."""
    result = dict(config)
    result["detector_mode"] = "Automatic"
    result["model"] = config.get("general_model", "yolo11s.pt")
    general = list(config.get("general_classes", GENERAL_CLASSES))
    result["remember_classes"] = list(dict.fromkeys(general + ["bottle", "moisturizer", "book"]))
    result["combined_detector"] = dict(config.get("combined_detector", {}), enabled=True)
    result["automatic_detector"] = dict(config.get("automatic_detector", {}), enabled=True)
    return result


def merge_sightings(confirmed, possible, session, query):
    """Current-session chronology wins over confidence; preserve both states.

    Never compare video seconds across different sessions. Multiple tracks are
    not assumed to be the same physical object. No memory record is modified.
    """
    merged = []
    latest_confirmed = {}
    for original in confirmed:
        row = dict(original)
        row["_possible"] = False
        row["current_session"] = row.get("session") == session
        latest_confirmed[(row.get("session"), row.get("id"))] = max(
            float(row.get("seconds", 0)),
            latest_confirmed.get((row.get("session"), row.get("id")), -1),
        )
        merged.append(row)
    generic = query.strip().casefold()
    for original in sorted(possible, key=lambda r: float(r.get("seconds", 0)), reverse=True):
        if not original.get("evidence"):
            continue
        if original.get("session", session) != session:
            continue
        tid = original["track"]
        seconds = float(original.get("seconds", 0))
        if seconds <= latest_confirmed.get((session, tid), -1) + .001:
            continue
        # Keep only the newest evidence per tentative track.
        if any(r.get("_possible") and r.get("id") == tid for r in merged):
            continue
        detector = original.get("detector_name", "")
        name = detector if generic == detector.casefold() else (original.get("candidate_name") or detector)
        row = dict(original)
        row.update(_possible=True, id=tid, name=name, nickname="", session=session,
                   current_session=True, last_seen=original.get("stamp", "Unknown"),
                   snapshot=original["evidence"], location="", personal_id="")
        merged.append(row)
    # Historical confirmed rows preserve engine.search's established session order.
    current = [r for r in merged if r.get("current_session")]
    historical = [r for r in merged if not r.get("current_session")]
    current.sort(key=lambda r: (float(r.get("seconds", 0)), not r["_possible"]), reverse=True)
    return current + historical


class App(RecognitionApp):
    def __init__(self, root):
        self.overlay = PreviewOverlay()
        self.detection_preview = None
        self.aligned_preview = tk.BooleanVar(master=root, value=True)
        self.tentative_results = []
        self.evidence_frame = None
        self.resize_job = None

        self.position_text = tk.StringVar(
            master=root,
            value="Last seen position: —",
        )

        super().__init__(root)
        self.best_picture.set(False)

        root.title(
            "Object Memory Assistant — Full Scene Evidence"
        )

    def build(self):
        super().build()
        toolbar = self.start_buttons[0].master
        self.mode_note = tk.StringVar(master=self.root)
        ttk.Label(toolbar.master, textvariable=self.mode_note, wraplength=900).pack(anchor="w", pady=(4, 0))
        ttk.Checkbutton(toolbar.master,
                        text="Detection view: aligned boxes on analysed frames (may be delayed)",
                        variable=self.aligned_preview).pack(anchor="w")
        self.update_mode_note()
        parent = self.picture.master

        for widget in parent.winfo_children():
            if (
                isinstance(widget, ttk.Label)
                and str(widget.cget("textvariable")) == str(self.answer)
            ):
                widget.grid_configure(row=4)

        self.picture.grid_configure(row=5)
        parent.rowconfigure(4, weight=0)
        parent.rowconfigure(5, weight=1)

        label = ttk.Label(
            parent,
            textvariable=self.position_text,
            font=("Segoe UI", 11, "bold"),
            wraplength=300,
        )
        label.grid(
            row=3, column=0, sticky="ew", pady=8
        )

        parent.bind(
            "<Configure>",
            lambda event: label.configure(
                wraplength=max(180, event.width - 24)
            ),
            add="+",
        )

        self.picture.bind(
            "<Configure>", self.resize_evidence
        )
        self.picture.bind(
            "<Double-1>", self.expand_evidence
        )

    def resize_evidence(self, event=None):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)

        self.resize_job = self.root.after(
            100, self.draw_evidence
        )

    def draw_evidence(self):
        self.resize_job = None

        if self.evidence_frame is not None:
            self.show_image(
                self.picture,
                self.evidence_frame,
                (
                    max(1, self.picture.winfo_width() - 4),
                    max(1, self.picture.winfo_height() - 4),
                ),
            )

    def expand_evidence(self, event=None):
        if self.evidence_frame is None:
            return

        window = tk.Toplevel(self.root)
        window.title(
            "Saved scene — "
            + self.position_text.get().split("\n")[0]
        )

        width = min(
            1100, self.root.winfo_screenwidth() - 80
        )
        height = min(
            760, self.root.winfo_screenheight() - 100
        )
        window.geometry(f"{width}x{height}")

        ttk.Label(
            window,
            text=self.position_text.get(),
            wraplength=950,
            padding=8,
        ).pack(fill="x")

        picture = ttk.Label(
            window, anchor="center"
        )
        picture.pack(fill="both", expand=True)

        frame = self.evidence_frame.copy()

        picture.bind(
            "<Configure>",
            lambda event: self.show_image(
                picture,
                frame,
                (
                    max(1, event.width - 4),
                    max(1, event.height - 4),
                ),
            ),
        )

    def annotate(self, frame, seconds):
        output = frame.copy()
        height, width = output.shape[:2]

        for tid, track in self.engine.tracks.items():
            if (
                abs(track["last_detection"] - seconds) > 0.001
                or track.get("rejected")
            ):
                continue

            name, color = display_identity(
                track, self.engine.records.get(tid)
            )

            if not valid_box(track["box"]):
                continue

            a, b, c, d = [
                int(value * dimension)
                for value, dimension in zip(
                    track["box"],
                    (width, height, width, height),
                )
            ]

            cv2.rectangle(
                output, (a, b), (c, d), color, 2
            )
            cv2.putText(
                output,
                f"#{tid} {name}",
                (a, max(18, b - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return output

    def on_frame_observed(
        self, frame, detections, seconds, stamp
    ):
        super().on_frame_observed(
            frame, detections, seconds, stamp
        )

        if getattr(self, "is_video", False):
            return

        # Retain the actual analysed frame, never paint its old coordinates
        # onto a newer live frame. Cache is tied to the source token.
        self.detection_preview = (self.token, self.annotate(frame, seconds), stamp, len(detections))

        entries = []

        for tid, track in self.engine.tracks.items():
            if (
                abs(track["last_detection"] - seconds) > 0.001
                or track.get("rejected")
            ):
                continue

            name, color = display_identity(
                track, self.engine.records.get(tid)
            )

            if valid_box(track["box"]):
                entries.append(
                    (track["box"], f"#{tid} {name}", color)
                )

        try:
            age = max(
                0,
                time.time()
                - datetime.fromisoformat(stamp).timestamp(),
            )
        except ValueError:
            age = 0

        try:
            self.overlay.seed(frame, entries, age)
        except (cv2.error, ValueError):
            self.overlay = PreviewOverlay()

    def show_image(self, widget, frame, size):
        if (widget is self.preview and self.running
                and not getattr(self, "is_video", False)
                and self.aligned_preview.get()):
            cached = self.detection_preview
            if cached and cached[0] == self.token:
                frame = cached[1].copy()
                try:
                    age = max(0, time.time() - datetime.fromisoformat(cached[2]).timestamp())
                    delay = f" | captured {age:.1f}s ago"
                except (ValueError, TypeError):
                    delay = ""
                message = f"ANALYSED FRAME{delay} | {cached[3]} detections"
            else:
                frame = frame.copy()
                message = "LIVE CAMERA | waiting for first detection result"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (25, 25, 25), -1)
            cv2.putText(frame, message, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (255, 255, 255), 1)
            return super().show_image(widget, frame, size)
        if (
            widget is self.preview
            and self.running
            and not getattr(self, "is_video", False)
        ):
            try:
                frame = self.overlay.render(frame)
            except (cv2.error, ValueError):
                self.overlay = PreviewOverlay()

        super().show_image(widget, frame, size)

    def update_mode_note(self):
        self.mode_note.set("Automatic recognition: standard objects + trained bottle/moisturizer/book. "
                           "Specialists check every second analysed frame. Personal names require review.")

    def process_frame(self, pipeline, frame, detections, seconds):
        # Provenance comes from the detector, not a UI mode or guessed class.
        tagged = [dict(d, category_source=d.get("category_source", "general")) for d in detections]
        return super().process_frame(pipeline, frame, tagged, seconds)

    def refresh_table(self):
        super().refresh_table()
        if not self.config.get("strict_personal_names", False):
            return
        for tid, track in self.engine.tracks.items():
            if not self.table.exists(str(tid)):
                continue
            record = self.engine.records.get(tid)
            match = track.get("suggestion", {})
            values = list(self.table.item(str(tid), "values"))
            if record and record.get("category_only") and not track.get("personal_id"):
                values[2] = record["name"] + " (category only)"
                values[3] = "Category observed; identity unverified"
                if match.get("display_allowed") and match.get("name"):
                    values[2] += " / Possible: " + match["name"]
                    values[3] = "Review personal identity"
            elif not match.get("display_allowed", False) and not track.get("personal_id"):
                values[2] = "—"
                values[3] = ("Category/identity conflict" if match.get("category_conflict")
                             else "No verified personal identity")
            if track.get("model_disputed"):
                values[2] = "—"
                values[3] = "Models disagree; memory update withheld"
            self.table.item(str(tid), values=values)

    def settings(self):
        return detector_profile(super().settings())

    def find(self):
        self.persist()
        self.evidence_frame = None
        self.tentative_results = []
        query = self.target.get().strip()
        aliases = {"phone": "cell phone", "waterbottle": "bottle",
                   "water bottle": "bottle", "notebook": "book"}
        registered_names = {item["name"].casefold() for item in self.registry.active()}
        if query.casefold() not in registered_names:
            query = aliases.get(query.casefold(), query)
        confirmed = self.engine.search(query)
        for row in confirmed:
            if row.get("current_session") and row["id"] in self.engine.records:
                row.update(self.engine.records[row["id"]])
        active = {item["id"] for item in self.registry.active()}
        excluded = {tid: set(t.get("denied_objects", set())) for tid, t in self.engine.tracks.items()}
        rejected = {tid for tid, t in self.engine.tracks.items() if t.get("rejected")}
        rows = [r for r in self.engine.history(False)
                if not r.get("candidate_id") or r["candidate_id"] in active]
        if self.config.get("strict_personal_names", False):
            safe_rows = []
            for original in rows:
                row = dict(original)
                track = self.engine.tracks.get(row["track"], {})
                match = track.get("suggestion", {})
                visible = (not track.get("model_disputed", False)
                           and row.get("strong") and match.get("display_allowed", False)
                           and row.get("candidate_id") == match.get("object_id"))
                if not visible:
                    # Keep weak hypotheses available to manual track review,
                    # but do not expose their personal names as search results.
                    row["candidate_id"] = ""
                    row["candidate_name"] = ""
                category_ok = (not track.get("model_disputed", False)
                               and track.get("category_ready", False)
                               and row.get("detector_confidence", 0) >= self.config.get("possible_category_confidence", .5))
                if not category_ok:
                    row["detector_name"] = ""
                safe_rows.append(row)
            rows = safe_rows
        possible = possible_sightings(rows, query, excluded, rejected)
        self.tentative_results = possible
        self.search_results = merge_sightings(confirmed, possible, self.engine.session, query)
        if not self.search_results:
            self.result_picker.configure(values=[])
            self.result_choice.set("")
            self.position_text.set("No matching confirmed or possible sighting.")
            self.answer.set("No matching saved evidence. A detector prediction is not automatically a confirmed identity.")
            self.picture.configure(image="", text="No matching saved scene")
            self.picture.image = None
            return
        choices = []
        for row in self.search_results:
            state = "POSSIBLE — UNCONFIRMED" if row["_possible"] else "CONFIRMED"
            scope = "this session" if row["current_session"] else "earlier session"
            choices.append(f"{state} | {row.get('nickname') or row['name']} | {row['last_seen']} | #{row['id']} | {scope}")
        self.result_picker.configure(values=choices)
        self.result_picker.current(0)
        self.show_result()

    def show_result(self):
        index = self.result_picker.current()
        if index < 0 or index >= len(self.search_results):
            return
        record = dict(self.search_results[index])
        possible = record.get("_possible", False)
        record["current_session"] = record.get("session") == self.engine.session
        if not possible and record.get("current_session") and record["id"] in self.engine.records:
            record.update(self.engine.records[record["id"]])
        self.render_evidence(record, possible)

    def closeup(self):
        # Keep full-scene context and the currently selected evidence timestamp.
        self.expand_evidence()

    def render_evidence(self, record, possible):
        self.persist()

        heading = (
            "LAST POSSIBLE POSITION — UNCONFIRMED"
            if possible
            else ("LAST OBSERVED CATEGORY POSITION — PERSONAL IDENTITY UNVERIFIED"
                  if record.get("category_only") and not record.get("personal_id")
                  else "LAST CONFIRMED POSITION")
        )
        legacy_auto = (not possible and not record.get("current_session")
                       and record.get("personal_id") and not record.get("user_labelled"))
        if legacy_auto:
            heading = "EARLIER AUTOMATIC MATCH — NOT REVALIDATED"

        self.position_text.set(
            heading
            + "\n"
            + position_description(record)
            + "\nSeen: "
            + record.get("last_seen", "Unknown")
        )

        name = record.get("nickname") or record["name"]

        message = (
            "Identity needs confirmation. Pause, select its row "
            "in Seen objects, then Confirm identity."
            if possible
            else (
                "Saved last-seen evidence; current physical "
                "location may have changed."
            )
        )

        if record.get("category_only") and not record.get("personal_id"):
            message = "Repeated category observations only. This is not confirmation of a particular registered object."
        if legacy_auto:
            message = "Saved by the earlier automatic naming policy. This identity has not been revalidated by the new checks; review it before relying on it."
        if "Models disagree" in record.get("status", ""):
            message += "\nNewer model predictions disagree. This is the older saved position; memory updates are currently withheld."
        if possible:
            reason = record.get("reason") or "This newer observation has not passed the memory/identity confirmation checks."
            message += "\nReason: " + reason
            older = next((r for r in getattr(self, "search_results", [])
                          if not r.get("_possible") and r.get("session") == record.get("session")
                          and float(r.get("seconds", 0)) < float(record.get("seconds", 0))), None)
            if older:
                message += "\nAn older CONFIRMED match is available in the result selector at " + older["last_seen"] + "."
            message += "\nMatching tracks may belong to different objects; this does not establish personal identity."
        if getattr(self, "is_video", False) and record.get("current_session"):
            gap = max(0, self.last_seconds - float(record.get("seconds", 0)))
            if gap > 1:
                message += f"\nThis evidence is {gap:.1f}s before the displayed video frame; it is not the final-frame position."
        if getattr(getattr(self, "engine", None), "camera_warning", False):
            message += "\nCamera view shifted: image zones may no longer describe current positions."

        self.answer.set(
            name
            + "\n"
            + message
            + "\nDouble-click the image to view the full scene larger."
        )

        source = record.get("snapshot", "")
        path = ROOT / source if source else None

        frame = (
            cv2.imread(str(path))
            if path and path.is_file()
            else None
        )

        self.evidence_frame = None

        if frame is None:
            self.picture.configure(
                image="",
                text="Saved scene unavailable or expired.",
            )
            self.picture.image = None
            return

        box = record.get("box")

        if valid_box(box):
            height, width = frame.shape[:2]

            a, b, c, d = [
                int(value * dimension)
                for value, dimension in zip(
                    box,
                    (width, height, width, height),
                )
            ]

            color = (
                (0, 160, 255)
                if possible
                else (0, 255, 255)
            )

            cv2.rectangle(
                frame, (a, b), (c, d), color, 3
            )
            cv2.putText(
                frame,
                ("Possible: " if possible else "") + name,
                (a, max(18, b - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Keep the ENTIRE scene. Do not crop to the object.
        self.evidence_frame = frame
        self.draw_evidence()

    def start(self, source, is_video):
        if (
            not self.worker_thread
            or not self.worker_thread.is_alive()
        ):
            self.overlay = PreviewOverlay()

        super().start(source, is_video)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
