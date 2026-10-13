"""Immediate replies for main-loop tests; native process behavior is tested separately."""

from pose.worker import PoseRead, PoseSnapshot


class ImmediatePoseStub:
    def __init__(self, estimator):
        self.estimator = estimator
        self.pending = None

    def submit(self, packet):
        self.pending = packet
        return True

    def read_result(self):
        if self.pending is None:
            return PoseRead('waiting')
        packet, self.pending = self.pending, None
        result = self.estimator.process_frame(packet.image, packet.timestamp_ms)
        return PoseRead('result', PoseSnapshot(packet, self.estimator.extract_landmarks(result), [], 0))

    def close(self):
        self.estimator.close()
