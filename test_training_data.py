import csv
import os
import tempfile
import unittest

from training_data import delete_label_samples


class DeleteLabelSamplesTests(unittest.TestCase):
    def test_deletes_only_matching_label_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "sign_data.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as data_file:
                writer = csv.writer(data_file)
                writer.writerows([
                    [1, 2, "A"],
                    [3, 4, "B"],
                    [5, 6, "a"],
                ])

            removed = delete_label_samples(csv_path, " A ")

            with open(csv_path, newline="", encoding="utf-8") as data_file:
                remaining = list(csv.reader(data_file))

        self.assertEqual(removed, 2)
        self.assertEqual(remaining, [["3", "4", "B"]])

    def test_rejects_empty_label_without_changing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "sign_data.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as data_file:
                data_file.write("1,2,A\n")

            with self.assertRaises(ValueError):
                delete_label_samples(csv_path, " ")

            with open(csv_path, encoding="utf-8") as data_file:
                contents = data_file.read()

        self.assertEqual(contents, "1,2,A\n")


if __name__ == "__main__":
    unittest.main()
