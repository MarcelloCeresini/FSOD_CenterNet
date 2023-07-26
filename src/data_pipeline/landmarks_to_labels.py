
from dataset_config import DatasetConfig

class LandmarksToLabels:
    def __init__(self,
                 conf: DatasetConfig) -> None:
        
        # model outputs (out_reg, out_heat_base, out_heat_novel)
        self.regressor_label_size = [4, *conf.input_to_model_resolution]

