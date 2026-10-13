import json
from pathlib import Path
import unittest

from control.calibration import CalibrationProfile
from control.gesture_recognizer import GestureRecognizer
from control.delivery import simulated_delivery
from evaluation.replay import summarize_events


class RecordedGestureTests(unittest.TestCase):
    def test_recorded_jumps_and_standing_arm_motion(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/recorded_gestures.json').read_text())
        for clip in fixture['clips']:
            with self.subTest(clip=clip['file']):
                recognizer = GestureRecognizer()
                recognizer._apply_profile(CalibrationProfile(**fixture['profile']))
                events = []
                for row in clip['rows']:
                    actions = recognizer.recognize(row['features'], row['timestamp_ms'])
                    if actions:
                        recognizer.acknowledge_delivery(simulated_delivery(actions))
                    events.extend({'timestamp_ms': row['timestamp_ms'], 'action': action} for action in actions)
                expected = [{'action': event['action'],
                             'start_ms': round(event['start_frame'] * 1000 / clip['fps']),
                             'end_ms': round(event['end_frame'] * 1000 / clip['fps'])}
                            for event in clip['events']]
                metrics = summarize_events(events, expected)
                self.assertEqual(metrics['true_positive'], len(expected))
                self.assertEqual(metrics['false_negative'], 0)
                self.assertEqual(metrics['false_positive'], 0)


if __name__ == '__main__':
    unittest.main()
