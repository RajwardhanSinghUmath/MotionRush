import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.download import download_clip
from evaluation.replay import summarize_events


class EvaluationTests(unittest.TestCase):
    def test_existing_video_is_verified_without_network(self):
        data = b'example recording'
        clip = {'file': 'sample.avi', 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, patch('evaluation.download.urlopen') as network:
            Path(directory, clip['file']).write_bytes(data)
            self.assertEqual(download_clip(clip, directory), Path(directory, clip['file']))
            network.assert_not_called()

    def test_changed_recording_is_rejected(self):
        clip = {'file': 'sample.avi', 'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, clip['file']).write_bytes(b'bad!')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                download_clip(clip, directory)

    def test_manifest_cannot_write_outside_destination(self):
        with self.assertRaises(ValueError):
            download_clip({'file': '../escape.avi'}, Path('.'))

    def test_duplicate_and_wrong_actions_count_as_false_positives(self):
        events = [{'timestamp_ms': 120, 'action': 'Squat'},
                  {'timestamp_ms': 140, 'action': 'Squat'},
                  {'timestamp_ms': 200, 'action': 'Jump'}]
        expected = [{'start_ms': 100, 'end_ms': 150, 'action': 'Squat'},
                    {'start_ms': 300, 'end_ms': 400, 'action': 'Jump'}]
        metrics = summarize_events(events, expected)
        self.assertEqual((metrics['true_positive'], metrics['false_positive'], metrics['false_negative']), (1, 2, 1))
        self.assertEqual(metrics['precision'], 1 / 3)
        self.assertEqual(metrics['recall'], 0.5)

    def test_no_predictions_is_a_miss_not_a_perfect_score(self):
        metrics = summarize_events([], [{'start_ms': 0, 'end_ms': 100, 'action': 'Jump'}])
        self.assertIsNone(metrics['precision'])
        self.assertEqual(metrics['recall'], 0)

    def test_overlapping_annotation_windows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            summarize_events([], [
                {'start_ms': 0, 'end_ms': 100, 'action': 'Jump'},
                {'start_ms': 50, 'end_ms': 150, 'action': 'Jump'},
            ])


if __name__ == '__main__':
    unittest.main()
