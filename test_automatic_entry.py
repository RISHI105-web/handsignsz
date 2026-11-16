import unittest
from unittest.mock import patch

from automatic_entry import AutomaticEntry, REST_LABEL, STABLE_ENTRY_FRAMES
from test_forget_sign import application


class AutomaticEntryTests(unittest.TestCase):
    def setUp(self):
        self.entry = AutomaticEntry()

    def show(self, label):
        return [
            self.entry.observe(label)
            for _ in range(STABLE_ENTRY_FRAMES)
        ]

    def test_adds_letters_immediately_and_repeats_after_rest(self):
        self.assertEqual(self.show("A")[-1], "A")
        self.assertTrue(all(result is None for result in self.show("A")))
        self.assertEqual(self.show("B")[-1], "B")
        self.show(REST_LABEL)
        self.assertEqual(self.show("B")[-1], "B")

    def test_unstable_and_lost_hand_do_not_repeat(self):
        self.show(REST_LABEL)
        self.show("A")
        self.entry.observe("B")
        self.entry.observe(None)
        self.assertTrue(all(result is None for result in self.show("A")))
        self.entry.reset()
        self.assertEqual(self.show("B")[-1], "B")

    def test_motion_event_is_added_once_and_rest_rearms_it(self):
        self.assertEqual(self.entry.observe("J", motion=True), "J")
        self.assertIsNone(self.entry.observe("J", motion=True))
        self.entry.observe(REST_LABEL)
        self.entry.observe(REST_LABEL)
        self.assertEqual(self.entry.observe("J", motion=True), "J")

    def test_quick_rest_repeats_letter_but_one_frame_does_not(self):
        self.assertEqual(self.show("A")[-1], "A")
        self.entry.observe(REST_LABEL)
        self.assertTrue(all(result is None for result in self.show("A")))
        self.entry.observe(REST_LABEL)
        self.entry.observe(REST_LABEL)
        self.assertEqual(self.show("A")[-1], "A")


class AutomaticEntryIntegrationTests(unittest.TestCase):
    def test_translate_stream_adds_immediately_and_only_in_translate(self):
        class FakeModel:
            classes_ = [REST_LABEL, "A", "B"]

            def predict(self, features):
                return [{1: REST_LABEL, 2: "A", 3: "B"}[features[0][2]]]

        class FakeDetector:
            label = 1

            def findHands(self, image):
                if self.label is None:
                    return [], image
                landmarks = [[0, 0] for _ in range(21)]
                landmarks[1][0] = self.label
                return [{"lmList": landmarks}], image

        class FakeMotion:
            def update(self, *args):
                return None

            def reset(self):
                pass

        class FakeBuffer:
            def tobytes(self):
                return b"frame"

        detector = FakeDetector()
        with patch.multiple(
            application, model=FakeModel(), detector=detector,
            automatic_entry=AutomaticEntry(), sentence="", current_prediction="",
            predictions=[], is_training=False, dynamic_prediction_frames=0,
            motion_recognizer=FakeMotion(), tensorflow_motion=FakeMotion(),
            motion_state={"labels": []},
        ), patch.object(application.cap, "read", return_value=(True, object())), \
                patch.object(application.cv2, "putText"), \
                patch.object(application.cv2, "imencode", return_value=(True, FakeBuffer())):
            stream = application.generate_frames(translate=True)
            for label, frames in [(2, 6), (2, 6), (3, 6), (1, 2), (3, 6)]:
                detector.label = label
                for _ in range(frames):
                    next(stream)
            self.assertEqual(application.sentence, "ABB")

            other_stream = application.generate_frames(translate=False)
            detector.label = 2
            for _ in range(STABLE_ENTRY_FRAMES):
                next(other_stream)
            self.assertEqual(application.sentence, "ABB")

            detector.label = None
            next(stream)
            detector.label = 2
            for _ in range(STABLE_ENTRY_FRAMES):
                next(stream)
            self.assertEqual(application.sentence, "ABBA")

    def test_rest_position_is_static_and_not_manually_added(self):
        with patch.multiple(application, is_training=False, current_prediction=REST_LABEL,
                            sentence="", automatic_entry=AutomaticEntry()):
            client = application.app.test_client()
            response = client.post(
                "/start_training", json={"label": REST_LABEL, "motion": True}
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(client.get("/add_letter").get_json()["sentence"], "")
            response = client.post(
                "/start_training", json={"label": REST_LABEL, "motion": False}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(application.training_label, REST_LABEL)


if __name__ == "__main__":
    unittest.main()
