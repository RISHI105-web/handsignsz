import csv
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from hand_features import isl_features, isl_motion_features
from tensorflow_motion import (
    TensorFlowMotionRecognizer, fit_motion, load_motion_state, prepare_sequence,
    save_motion_state,
)
from test_forget_sign import application
from training_data import fit_model, load_model_state, save_model_state


def hand(x, y):
    landmarks = [[x + index, y + index] for index in range(21)]
    return {"lmList": landmarks}


class LanguageModeTests(unittest.TestCase):
    def test_two_hand_features_are_order_independent_and_include_both_hands(self):
        left, right = hand(100, 100), hand(200, 150)
        first = isl_features([left, right])
        self.assertEqual(len(first), 85)
        self.assertEqual(first, isl_features([right, left]))
        self.assertEqual(first[42:44], [100, 50])
        self.assertEqual(first[-1], 2)
        self.assertEqual(isl_features([left])[-1], 1)

    def test_two_hand_motion_sequence_is_supported(self):
        frames = [
            isl_motion_features([hand(100 + index * 2, 100), hand(200, 150)])
            for index in range(30)
        ]
        self.assertEqual(prepare_sequence(frames).shape, (30, 87))

    def test_two_hand_tensorflow_model_trains_and_reloads(self):
        recordings = []
        for offset in range(3):
            recordings.append({
                "label": "B",
                "frames": [
                    isl_motion_features([
                        hand(100 + index * 2 + offset, 100),
                        hand(200 + offset, 150),
                    ])
                    for index in range(30)
                ],
            })
        state = fit_motion(recordings)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "motion.pkl")
            save_motion_state(path, state)
            recognizer = TensorFlowMotionRecognizer(load_motion_state(path), isl_motion_features)
            results = [
                recognizer.update([hand(100 + index * 2, 100), hand(200, 150)])
                for index in range(30)
            ]
        self.assertEqual(results[-1], "B")

    def test_mode_navigation_training_data_and_images_are_separate(self):
        class FakeDetector:
            def findHands(self, image):
                return [hand(100, 100), hand(200, 150)], image

        class FakeBuffer:
            def tobytes(self):
                return b"frame"

        old_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                os.mkdir("models")
                os.mkdir("signpictures")
                os.mkdir("signpictures_isl")
                with open(os.path.join("signpictures", "A.jpg"), "wb") as file:
                    file.write(b"ASL")
                with open(os.path.join("signpictures_isl", "A.jpg"), "wb") as file:
                    file.write(b"ISL")
                with open("sign_data.csv", "w", newline="") as file:
                    csv.writer(file).writerow([1] * 42 + ["A"])
                with patch.multiple(
                    application, active_language="asl", mode_generation=0,
                    MODEL_PATH="models/sign_language_model.pkl",
                    DATA_PATH="sign_data.csv", MOTION_PATH="models/motion_data.pkl",
                    model=None, disabled_motion=set(), is_training=False,
                    motion_state={"recordings": [], "labels": [], "weights": None},
                    tensorflow_motion=application.TensorFlowMotionRecognizer(
                        {"recordings": [], "labels": [], "weights": None}
                    ),
                    detector=FakeDetector(), sentence="", current_prediction="",
                    predictions=[], training_motion_frames=[],
                ), patch.object(application.cap, "read", return_value=(True, object())), \
                        patch.object(application.cv2, "putText"), \
                        patch.object(application.cv2, "imencode", return_value=(True, FakeBuffer())):
                    client = application.app.test_client()
                    self.assertIn(b"ISL (two-handed)", client.get("/").data)
                    self.assertIn(b"Pre-trained ISL", client.get("/pretrained").data)
                    self.assertIn(b"ISL Sign Language Translator", client.get("/app/isl").data)
                    self.assertEqual(client.get("/sign_image/A?language=isl").data, b"ISL")
                    self.assertEqual(client.get("/sign_image/A?language=asl").status_code, 409)

                    response = client.post(
                        "/start_training?language=isl", json={"label": "B", "motion": False}
                    )
                    self.assertEqual(response.status_code, 200)
                    stream = application.generate_frames(generation=application.mode_generation)
                    for _ in range(10):
                        next(stream)
                    self.assertEqual(client.get("/app/asl").status_code, 409)
                    self.assertEqual(client.post("/stop_training?language=isl").status_code, 200)
                    with open("isl_sign_data.csv", newline="") as file:
                        rows = list(csv.reader(file))
                    self.assertEqual(len(rows), 10)
                    self.assertEqual(len(rows[0]), 86)
                    with open("sign_data.csv", newline="") as file:
                        asl_rows = list(csv.reader(file))
                    self.assertEqual(asl_rows[0][-1], "A")
                    self.assertEqual(len(asl_rows[0]), 43)

                    self.assertEqual(client.post("/train_model?language=isl").get_json()["status"], "success")
                    self.assertTrue(os.path.isfile("models/isl_sign_language_model.pkl"))
                    self.assertIn(b"ASL Sign Language Translator", client.get("/app/asl").data)
                    with self.assertRaises(StopIteration):
                        next(stream)
                    self.assertEqual(client.get("/sign_image/A?language=asl").data, b"ASL")
                    self.assertEqual(client.get("/sentence?language=isl").status_code, 409)
            finally:
                os.chdir(old_directory)

    def test_pretrained_pages_use_separate_models_images_and_block_training(self):
        old_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                os.mkdir("models")
                for language in ("asl", "isl"):
                    image_dir = f"signpictures_pretrained_{language}"
                    os.mkdir(image_dir)
                    with open(os.path.join(image_dir, "A.jpg"), "wb") as file:
                        file.write(language.encode())
                empty_state = {"recordings": [], "labels": [], "weights": None}
                with patch.multiple(
                    application, active_language="asl", mode_generation=0,
                    MODEL_PATH="models/sign_language_model.pkl",
                    DATA_PATH="sign_data.csv", MOTION_PATH="models/motion_data.pkl",
                    model=None, disabled_motion=set(), is_training=False,
                    load_model_state=load_model_state,
                    motion_state=empty_state,
                    tensorflow_motion=TensorFlowMotionRecognizer(empty_state),
                    sentence="", current_prediction="", predictions=[],
                ):
                    client = application.app.test_client()
                    choices = client.get("/pretrained").get_data(as_text=True)
                    self.assertIn('/app/pretrained_asl', choices)
                    self.assertIn('/app/pretrained_isl', choices)
                    for language in ("asl", "isl"):
                        mode = f"pretrained_{language}"
                        page = client.get(f"/app/{mode}").get_data(as_text=True)
                        self.assertIn(f"Pre-trained {language.upper()} Sign Language Translator", page)
                        self.assertIn("Speech to Sign", page)
                        self.assertIn("Translate", page)
                        self.assertIn("Supported recognition vocabulary", page)
                        self.assertIn("No signs available until a compatible model is installed.", page)
                        self.assertIn("Recognized sign/text:", page)
                        self.assertNotIn('id="train"', page)
                        self.assertNotIn("switchTab('train')", page)
                        self.assertEqual(
                            client.get(f"/sign_image/A?language={mode}").data,
                            language.encode(),
                        )
                        self.assertFalse(client.get(f"/current_sign?language={mode}").get_json()["model_available"])
                        for endpoint in ("start_training", "stop_training", "train_model",
                                         "delete_training_data"):
                            self.assertEqual(client.post(f"/{endpoint}?language={mode}").status_code, 403)

                    for language, feature_count in (("asl", 42), ("isl", 85)):
                        data = pd.DataFrame([[1] * feature_count + [language.upper()]])
                        save_model_state(
                            f"models/pretrained_{language}_model.pkl", fit_model(data), set()
                        )
                    client.get("/app/asl")
                    for language, feature_count in (("asl", 42), ("isl", 85)):
                        mode = f"pretrained_{language}"
                        page = client.get(f"/app/{mode}").get_data(as_text=True)
                        self.assertEqual(application.model.n_features_in_, feature_count)
                        self.assertTrue(client.get(f"/current_sign?language={mode}").get_json()["model_available"])
                        self.assertRegex(
                            page,
                            rf"(?s)Supported recognition vocabulary:</strong>\s*{language.upper()}\s*</p>",
                        )
            finally:
                os.chdir(old_directory)


if __name__ == "__main__":
    unittest.main()
