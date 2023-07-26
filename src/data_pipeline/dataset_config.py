
class DatasetConfig:
    
    def __init__(self) -> None:
        self.input_to_model_resolution = (512, 512)
        self.crop_scale = (0.5, 1.0)
        self.crop_ratio = (0.5, 1.5)

        self.p_vertical_flip = 0.5
        self.p_horizontal_flip = 0.5

        