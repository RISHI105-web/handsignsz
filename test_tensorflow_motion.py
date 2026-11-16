import os
import tempfile
import unittest
from unittest.mock import patch

from tensorflow_motion import (
    TensorFlowMotionRecognizer, fit_motion, frame_features, load_motion_state,
    prepare_sequence, save_motion_state,
)


def landmarks_for_frame(index, axis, offset=0):
    landmarks = [[100, 200] for _ in range(21)]
    landmarks[5] = [150, 150]
    landmarks[17] = [250, 150]
    distance = index * 3 + offset
    for point in landmarks:
        point[axis] += distance
    return landmarks


class TensorFlowMotionTests(unittest.TestCase):
    def test_short_recording_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_sequence([[0] * 44] * 9)

    def test_requires_multiple_recordings(self):
        recording = {
            "label": "J",
            "frames": [frame_features(landmarks_for_frame(i, 0)) for i in range(30)],
        }
        with self.assertRaisesRegex(ValueError, "at least 3"):
            fit_motion([recording])

    def test_train_save_reload_and_recognize_motion(self):
        recordings = []
        for label, axis in [("J", 0), ("Z", 1)]:
            for offset in range(3):
                recordings.append({
                    "label": label,
                    "frames": [
                        frame_features(landmarks_for_frame(i, axis, offset))
                        for i in range(30)
                    ],
                })
        state = fit_motion(recordings)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "motion.pkl")
            save_motion_state(path, state)
            recognizer = TensorFlowMotionRecognizer(load_motion_state(path))

            results = [
                recognizer.update(landmarks_for_frame(i, 0))
                for i in range(30)
            ]
            self.assertEqual(results[-1], "J")
            recognizer.reset()
            results = [
                recognizer.update(landmarks_for_frame(0, 0))
                for _ in range(30)
            ]
            self.assertTrue(all(result is None for result in results))

    def test_record_train_and_forget_motion_via_routes(self):
        from test_forget_sign import application

        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "data.csv")
            model_path = os.path.join(directory, "model.pkl")
            motion_path = os.path.join(directory, "motion.pkl")
            empty_state = {"recordings": [], "labels": [], "weights": None}
            with patch.multiple(
                application, DATA_PATH=data_path, MODEL_PATH=model_path,
                MOTION_PATH=motion_path, model=None, disabled_motion=set(),
                motion_state=empty_state,
                tensorflow_motion=TensorFlowMotionRecognizer(empty_state),
                is_training=False, training_motion=False,
                training_count=0, training_label="", training_motion_frames=[],
            ):
                client = application.app.test_client()
                client.post("/start_training", json={"label": "J", "motion": True})
                application.training_motion_frames.extend([
                    frame_features(landmarks_for_frame(0, 0))
                    for _ in range(30)
                ])
                response = client.post("/stop_training")
                self.assertEqual(response.status_code, 400)
                self.assertIn("No motion detected", response.get_json()["message"])
                for offset in range(3):
                    response = client.post(
                        "/start_training", json={"label": "j", "motion": True}
                    )
                    self.assertEqual(response.status_code, 200)
                    application.training_motion_frames.extend([
                        frame_features(landmarks_for_frame(i, 0, offset))
                        for i in range(30)
                    ])
                    application.training_count = 30
                    response = client.post("/stop_training")
                    self.assertEqual(response.get_json()["samples"], 1)

                response = client.post("/train_model")
                self.assertEqual(response.get_json()["status"], "success")
                self.assertIn("J", load_motion_state(motion_path)["labels"])

                response = client.post("/delete_training_data", json={"label": "J"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["removed_motion"], 3)
                self.assertEqual(load_motion_state(motion_path)["recordings"], [])
                self.assertNotIn("J", load_motion_state(motion_path)["labels"])
                self.assertIn("J", application.disabled_motion)


if __name__ == "__main__":
    unittest.main()
