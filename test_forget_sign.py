import csv
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from training_data import fit_model, load_model_state, save_model_state
from automatic_entry import REST_LABEL, STABLE_ENTRY_FRAMES


with patch("training_data.load_model_state", return_value=(None, set())), \
        patch("cv2.VideoCapture"), \
        patch("cvzone.HandTrackingModule.HandDetector"):
    import app as application


class ForgetSignTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.data_path = os.path.join(self.directory.name, "data.csv")
        self.model_path = os.path.join(self.directory.name, "model.pkl")
        self.paths = patch.multiple(
            application, DATA_PATH=self.data_path, MODEL_PATH=self.model_path,
            MOTION_PATH=os.path.join(self.directory.name, "motion.pkl")
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)
        application.is_training = False
        application.current_prediction = ""
        application.predictions.clear()
        application.disabled_motion = set()
        application.motion_state = {"recordings": [], "labels": [], "weights": None}
        application.tensorflow_motion = application.TensorFlowMotionRecognizer(application.motion_state)
        self.client = application.app.test_client()

    def write_samples(self, labels):
        with open(self.data_path, "w", newline="") as file:
            writer = csv.writer(file)
            for index, label in enumerate(labels):
                writer.writerow([index] * 42 + [label])
        samples = pd.read_csv(self.data_path, header=None)
        application.model = fit_model(samples)
        save_model_state(self.model_path, application.model, set())

    def test_forget_rebuilds_saved_model_and_clears_prediction(self):
        self.write_samples(["A", "A", "B", "B"])
        application.current_prediction = "A"
        application.predictions.extend(["A", "A"])

        response = self.client.post("/delete_training_data", json={"label": "a"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["removed"], 2)
        self.assertEqual(application.current_prediction, "")
        self.assertEqual(application.predictions, [])
        restored, _ = load_model_state(self.model_path)
        self.assertEqual(list(restored.classes_), ["B"])
        self.assertEqual(list(application.model.classes_), ["B"])
        self.assertEqual(set(pd.read_csv(self.data_path, header=None).iloc[:, -1]), {"B"})

    def test_forgetting_last_class_leaves_no_static_model(self):
        self.write_samples(["A", "A"])

        response = self.client.post("/delete_training_data", json={"label": "A"})

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(application.model)
        restored, _ = load_model_state(self.model_path)
        self.assertIsNone(restored)

    def test_motion_is_disabled_until_new_samples_are_trained(self):
        self.write_samples(["A", "A", "A", "A", "J", "J"])

        response = self.client.post("/delete_training_data", json={"label": "J"})
        self.assertEqual(response.status_code, 200)
        restored, disabled = load_model_state(self.model_path)
        self.assertNotIn("J", restored.classes_)
        self.assertIn("J", disabled)

        response = self.client.post("/train_model")
        self.assertEqual(response.get_json()["status"], "success")
        self.assertIn("J", load_model_state(self.model_path)[1])

        with open(self.data_path, "a", newline="") as file:
            csv.writer(file).writerows([[10] * 42 + ["J"], [11] * 42 + ["J"]])
        self.assertIn("J", load_model_state(self.model_path)[1])

        response = self.client.post("/train_model")
        self.assertEqual(response.get_json()["status"], "success")
        restored, disabled = load_model_state(self.model_path)
        self.assertIn("J", restored.classes_)
        self.assertNotIn("J", disabled)

    def test_delete_while_recording_is_rejected(self):
        self.write_samples(["A", "B"])
        application.is_training = True
        self.addCleanup(setattr, application, "is_training", False)

        response = self.client.post("/delete_training_data", json={"label": "A"})

        self.assertEqual(response.status_code, 409)
        restored, _ = load_model_state(self.model_path)
        self.assertEqual(set(restored.classes_), {"A", "B"})

    def test_train_and_forget_rest_position(self):
        self.write_samples(["A", "A", "A", REST_LABEL, REST_LABEL, REST_LABEL])
        response = self.client.post("/train_model")
        self.assertEqual(response.get_json()["status"], "success")
        restored, _ = load_model_state(self.model_path)
        self.assertIn(REST_LABEL, restored.classes_)
        self.assertTrue(self.client.get("/current_sign").get_json()["rest_trained"])

        for _ in range(STABLE_ENTRY_FRAMES):
            application.automatic_entry.observe(REST_LABEL)
        response = self.client.post("/delete_training_data", json={"label": REST_LABEL})
        self.assertEqual(response.status_code, 200)
        restored, _ = load_model_state(self.model_path)
        self.assertNotIn(REST_LABEL, restored.classes_)
        status = self.client.get("/current_sign").get_json()
        self.assertFalse(status["rest_trained"])
        self.assertTrue(status["ready"])


if __name__ == "__main__":
    unittest.main()
