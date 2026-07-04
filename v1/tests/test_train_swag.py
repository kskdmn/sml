import inspect
import sys
import unittest
from unittest import mock
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


class TrainSwagTest(unittest.TestCase):
    def test_format_swag_example_concatenates_gold_ending(self):
        import train_swag

        row = {
            "startphrase": "A man is sitting on a roof. he",
            "ending0": " starts typing on a laptop.",
            "ending1": " is using wrap to blend a container.",
            "ending2": " is ripping level tiles off.",
            "ending3": " is holding a rubik's cube.",
            "label": 0,
        }

        self.assertEqual(
            "A man is sitting on a roof. he starts typing on a laptop.",
            train_swag.format_swag_example(row),
        )

    def test_resolve_swag_label_accepts_string_labels(self):
        import train_swag

        self.assertEqual(2, train_swag.resolve_swag_label("2"))

    def test_iter_swag_texts_resumes_after_saved_position(self):
        import train_swag
        from sml_config import SwagFineTuneConfig
        from train_sml import TrainingDataState

        rows = [
            {
                "startphrase": "first",
                "ending0": " one",
                "ending1": " two",
                "ending2": " three",
                "ending3": " four",
                "label": 0,
            },
            {
                "startphrase": "second",
                "ending0": " alpha",
                "ending1": " beta",
                "ending2": " gamma",
                "ending3": " delta",
                "label": 1,
            },
            {
                "startphrase": "third",
                "ending0": " x",
                "ending1": " y",
                "ending2": " z",
                "ending3": " w",
                "label": 2,
            },
        ]
        dataset = mock.Mock()
        dataset.__len__ = mock.Mock(return_value=len(rows))
        dataset.__getitem__ = mock.Mock(side_effect=lambda index: rows[index])
        data_state = TrainingDataState(line_number=0)

        with mock.patch.object(train_swag, "load_swag_dataset", return_value=dataset):
            texts = list(
                train_swag.iter_swag_texts(
                    SwagFineTuneConfig(shuffle_examples=False, seed=42),
                    epoch=0,
                    data_state=data_state,
                )
            )

        self.assertEqual(["second beta", "third z"], texts)
        self.assertEqual(2, data_state.line_number)

    def test_iter_swag_texts_updates_reading_progress_example_index(self):
        import train_swag
        from sml_config import SwagFineTuneConfig
        from train_sml import ReadingProgress

        rows = [
            {
                "startphrase": "first",
                "ending0": " one",
                "ending1": " two",
                "ending2": " three",
                "ending3": " four",
                "label": 0,
            },
            {
                "startphrase": "second",
                "ending0": " alpha",
                "ending1": " beta",
                "ending2": " gamma",
                "ending3": " delta",
                "label": 1,
            },
        ]
        dataset = mock.Mock()
        dataset.__len__ = mock.Mock(return_value=len(rows))
        dataset.__getitem__ = mock.Mock(side_effect=lambda index: rows[index])
        progress = ReadingProgress()

        with mock.patch.object(train_swag, "load_swag_dataset", return_value=dataset):
            texts = list(
                train_swag.iter_swag_texts(
                    SwagFineTuneConfig(shuffle_examples=False, seed=42),
                    epoch=0,
                    progress=progress,
                )
            )

        self.assertEqual(["first one", "second beta"], texts)
        self.assertEqual(1, progress.line_number)
        self.assertEqual(1, progress.example_index)

    def test_parse_args_defaults_to_fresh_fine_tuning(self):
        import train_swag

        args = train_swag.parse_args([])

        self.assertFalse(args.resume)

    def test_parse_args_enables_resume(self):
        import train_swag

        args = train_swag.parse_args(["--resume"])

        self.assertTrue(args.resume)

    def test_main_passes_resume_flag_to_fine_tune_swag(self):
        import train_swag

        with mock.patch.object(
            train_swag,
            "fine_tune_swag",
            return_value=Path("/tmp/sml-swag.pt"),
        ) as fine_tune_swag:
            return_code = train_swag.main(["--resume"])

        self.assertEqual(train_swag.SUCCESS_RETURN_CODE, return_code)
        self.assertTrue(fine_tune_swag.call_args.kwargs["resume_from_checkpoint"])

    def test_fine_tune_swag_accepts_config_objects_and_resume_flag(self):
        import train_swag

        parameters = inspect.signature(train_swag.fine_tune_swag).parameters

        self.assertEqual(
            ["fine_tune_config", "resume_from_checkpoint"],
            list(parameters),
        )


if __name__ == "__main__":
    unittest.main()
