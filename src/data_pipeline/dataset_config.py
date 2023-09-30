
class DatasetConfig:
    
    def __init__(self) -> None:
        self.input_to_model_resolution = (64, 64)
        self.output_resolution = (16, 16)
        self.output_stride = (self.input_to_model_resolution[0] // self.output_resolution[0],
                              self.input_to_model_resolution[1] // self.output_resolution[1])

        self.crop_scale = (0.5, 1.0)
        self.crop_ratio = (0.5, 1.5)

        self.p_vertical_flip = 0.5
        self.p_horizontal_flip = 0.5

        # sigma gaussian blur limits
        self.sgb_lims = (0.01, 2.0)

        self.min_IoU_for_gaussian_radius = 0.8 # from "CornerNet: Detecting Objects as Paired Keypoints", paragraph 3.2

        self.max_detections = 100 # TODO: check this value