import unittest

from motion_recognition import (
    MotionRecognizer,
    recognize_j_path,
    recognize_z_path,
)


def make_landmarks(letter, point):
    landmarks = [[100, 200] for _ in range(21)]
    landmarks[5] = [150, 150]
    landmarks[17] = [250, 150]

    if letter == "J":
        landmarks[6] = [150, 160]
        landmarks[8] = [145, 165]
        landmarks[18] = [200, 150]
        landmarks[20] = [
            100 + point[0] * 100,
            200 + point[1] * 100,
        ]
    else:
        landmarks[6] = [120, 180]
        landmarks[8] = [
            100 + point[0] * 100,
            200 + point[1] * 100,
        ]
        landmarks[18] = [150, 170]
        landmarks[20] = [145, 175]

    return landmarks


class MotionRecognitionTests(unittest.TestCase):
    def test_recognizes_j_downstroke_and_hook(self):
        path = [
            (0.0, index * 0.06)
            for index in range(9)
        ] + [
            (index * 0.06, 0.48)
            for index in range(1, 7)
        ]

        self.assertTrue(recognize_j_path(path))

    def test_recognizes_z_zigzag(self):
        path = (
            [(index * 0.08, 0.0) for index in range(9)]
            + [(0.64 - index * 0.06, index * 0.06) for index in range(1, 9)]
            + [(0.16 + index * 0.08, 0.48) for index in range(1, 9)]
        )

        self.assertTrue(recognize_z_path(path))

    def test_does_not_recognize_straight_motion_as_z(self):
        path = [(index * 0.08, index * 0.08) for index in range(12)]

        self.assertFalse(recognize_z_path(path))

    def test_motion_recognizer_tracks_j_pinky(self):
        recognizer = MotionRecognizer()
        path = (
            [(1.2, -0.3 + index * 0.1) for index in range(9)]
            + [(1.2 + index * 0.08, 0.5) for index in range(1, 7)]
        )

        for _ in range(3):
            recognizer.update(make_landmarks("J", path[0]))
        for point in path[1:]:
            result = recognizer.update(make_landmarks("J", point))
        for _ in range(7):
            result = recognizer.update(make_landmarks("J", path[-1]))

        self.assertEqual(result, "J")

    def test_disabled_motion_is_not_recognized(self):
        recognizer = MotionRecognizer()
        path = (
            [(1.2, -0.3 + index * 0.1) for index in range(9)]
            + [(1.2 + index * 0.08, 0.5) for index in range(1, 7)]
        )
        results = [
            recognizer.update(make_landmarks("J", point), {"J"})
            for point in path + [path[-1]] * 7
        ]
        self.assertNotIn("J", results)

    def test_motion_recognizer_tracks_z_index(self):
        recognizer = MotionRecognizer()
        path = (
            [(index * 0.1 + 0.2, -0.8) for index in range(11)]
            + [(1.2 - index * 0.07, -0.8 + index * 0.1) for index in range(1, 11)]
            + [(0.5 + index * 0.1, 0.2) for index in range(1, 11)]
        )

        for _ in range(3):
            recognizer.update(make_landmarks("Z", path[0]))
        for point in path[1:]:
            result = recognizer.update(make_landmarks("Z", point))
        for _ in range(7):
            result = recognizer.update(make_landmarks("Z", path[-1]))

        self.assertEqual(result, "Z")


if __name__ == "__main__":
    unittest.main()
