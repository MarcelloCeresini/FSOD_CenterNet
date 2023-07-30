class EvalConfig:
    def __init__(self):
        self.IoU_threshold_sets = {
            "50": [0.5],
            "75": [0.75],
            "50-95": [x / 100 for x in range(50, 100, 5)],
        }