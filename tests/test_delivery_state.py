import unittest

from control.calibration import CalibrationProfile
from control.delivery import DeliveryResult, simulated_delivery
from control.gesture_recognizer import GestureRecognizer


class DeliveryStateTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = GestureRecognizer()
        self.recognizer._apply_profile(CalibrationProfile(.5, .5, .25, .18, .05, .0875, .1375))

    def move(self, x=.5, y=.5, timestamp=4000):
        return self.recognizer.recognize({'hip_center': (x, y)}, timestamp)

    def test_lane_does_not_advance_until_delivery(self):
        actions = self.move(x=.8)
        self.assertEqual(actions, ['Right'])
        self.assertEqual(self.recognizer.current_lane, 'CENTER')
        self.assertEqual(self.move(x=.8, timestamp=4100), [])
        self.recognizer.acknowledge_delivery(simulated_delivery(actions))
        self.assertEqual(self.recognizer.current_lane, 'RIGHT')

    def test_partial_double_move_advances_one_lane_and_pauses(self):
        self.recognizer.reset_lane('LEFT')
        actions = self.move(x=.8)
        self.assertEqual(actions, ['Right', 'Right'])
        self.recognizer.acknowledge_delivery(DeliveryResult(tuple(actions), ('Right',), 'down failed', True))
        self.assertEqual(self.recognizer.current_lane, 'CENTER')
        self.assertTrue(self.recognizer.input_fault)
        self.assertTrue(self.recognizer.input_uncertain)
        self.assertEqual(self.move(x=.8, timestamp=5000), [])

    def test_failed_first_stroke_cannot_claim_lane_change(self):
        actions = self.move(x=.2)
        self.recognizer.acknowledge_delivery(DeliveryResult(tuple(actions), (), 'failed', True))
        self.assertEqual(self.recognizer.current_lane, 'CENTER')
        self.assertEqual(self.move(x=.2, timestamp=5000), [])

    def test_mixed_vertical_and_lane_failure_commits_only_jump(self):
        actions = self.move(x=.8, y=.3)
        self.assertEqual(actions, ['Jump', 'Right'])
        self.recognizer.acknowledge_delivery(DeliveryResult(tuple(actions), ('Jump',), 'failed', True))
        self.assertEqual(self.recognizer.current_lane, 'CENTER')
        self.assertEqual(self.recognizer.last_up_time, 4)
        self.assertEqual(self.recognizer.last_horizontal_time, float('-inf'))

    def test_reset_after_fault_requires_neutral_before_any_new_command(self):
        actions = self.move(y=.3)
        self.recognizer.acknowledge_delivery(DeliveryResult(tuple(actions), (), 'failed', True))
        self.recognizer.reset_lane()
        self.assertEqual(self.move(y=.3, timestamp=5000), [])
        self.assertEqual(self.move(timestamp=5100), [])
        self.assertEqual(self.move(y=.3, timestamp=5200), ['Jump'])

    def test_recalibration_and_tracking_loss_do_not_clear_input_fault(self):
        actions = self.move(x=.8)
        self.recognizer.acknowledge_delivery(DeliveryResult(tuple(actions), (), 'failed', True))
        self.recognizer.start_calibration()
        self.recognizer._apply_profile(CalibrationProfile(.5, .5, .25, .18, .05, .0875, .1375))
        self.assertEqual(self.move(x=.8, timestamp=5000), [])
        self.assertEqual(self.recognizer.recognize(None, 5100), [])
        self.assertTrue(self.recognizer.input_fault)

    def test_wrong_acknowledgement_pauses_without_applying_it(self):
        self.move(x=.8)
        with self.assertRaises(ValueError):
            self.recognizer.acknowledge_delivery(simulated_delivery(['Left']))
        self.assertTrue(self.recognizer.input_fault)
        self.assertEqual(self.recognizer.current_lane, 'CENTER')

    def test_delivery_cannot_report_a_nonprefix_as_completed(self):
        with self.assertRaises(ValueError):
            DeliveryResult(('Jump', 'Right'), ('Right',))


if __name__ == '__main__':
    unittest.main()
