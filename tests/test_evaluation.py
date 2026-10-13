import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from evaluation.download import download_clip
from evaluation.replay import summarize_events, score_motion
from evaluation.paired_replay import check_reference_alignment
from control.calibration import CalibrationProfile


class EvaluationTests(unittest.TestCase):
    def test_reference_from_different_standing_position_is_not_scored(self):
        profile = CalibrationProfile(0.5, 0.5, 0.25, 0.18, 0.05, 0.0875, 0.1375)
        features = {'hip_center': (0.7, 0.5), 'shoulder_center': (0.7, 0.25),
                    'torso_height': 0.25, 'shoulder_width': 0.18,
                    'confidence': 0.9, 'knee_angles': [175, 175]}
        alignment = check_reference_alignment(profile, [features, features])
        self.assertFalse(alignment['matched'])
        features['hip_center'] = (0.5, 0.5)
        features['shoulder_center'] = (0.5, 0.25)
        self.assertTrue(check_reference_alignment(profile, [features, features])['matched'])

    def test_disabled_or_unusable_controller_cannot_receive_accuracy_score(self):
        for result in ({'calibration_success': False, 'usable_ready_frames': 10},
                       {'calibration_success': True, 'usable_ready_frames': 0}):
            score = score_motion(result, [])
            self.assertFalse(score['eligible'])
            self.assertNotIn('metrics', score)

    def test_calibrated_replay_reports_a_missed_jump(self):
        result = {'calibration_success': True, 'usable_ready_frames': 40, 'events': []}
        score = score_motion(result, [{'action': 'Jump', 'start_ms': 100, 'end_ms': 300}])
        self.assertTrue(score['eligible'])
        self.assertEqual(score['metrics']['false_negative'], 1)
        self.assertEqual(score['metrics']['recall'], 0)

    def archive_clip(self, member='nested/sample.avi'):
        video = b'example recording'
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr(member, video)
        source = stream.getvalue()
        return video, source, {
            'file': 'sample.avi', 'bytes': len(video), 'sha256': hashlib.sha256(video).hexdigest(),
            'archive_member': member,
            'archive': {'file': 'source.zip', 'bytes': len(source),
                        'sha256': hashlib.sha256(source).hexdigest()},
        }

    def test_verified_archive_member_uses_only_destination_filename(self):
        video, source, clip = self.archive_clip('../outside.avi')
        with tempfile.TemporaryDirectory() as directory, patch('evaluation.download.urlopen') as network:
            archive_directory = Path(directory, '.archives')
            archive_directory.mkdir()
            (archive_directory / 'source.zip').write_bytes(source)
            target = download_clip(clip, directory)
            self.assertEqual(target, Path(directory, 'sample.avi'))
            self.assertEqual(target.read_bytes(), video)
            self.assertFalse(Path(directory, 'outside.avi').exists())
            network.assert_not_called()

    def test_corrupt_archive_is_rejected_before_extracting(self):
        video, source, clip = self.archive_clip()
        with tempfile.TemporaryDirectory() as directory:
            archive_directory = Path(directory, '.archives')
            archive_directory.mkdir()
            (archive_directory / 'source.zip').write_bytes(source + b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                download_clip(clip, directory)
            self.assertFalse(Path(directory, 'sample.avi').exists())

    def test_archive_member_hash_is_checked(self):
        video, source, clip = self.archive_clip()
        clip['sha256'] = hashlib.sha256(b'different').hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            archive_directory = Path(directory, '.archives')
            archive_directory.mkdir()
            (archive_directory / 'source.zip').write_bytes(source)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                download_clip(clip, directory)

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
