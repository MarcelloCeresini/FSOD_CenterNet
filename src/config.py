class Config:
    def __init__(self) -> None:

        # TODO: check if the following are ok

        self.train_batch_size = 64
        self.num_workers = 1
        self.pin_memory = True
        self.drop_last = False

        self.annotation_dict = {"base": {"train": "base_train.json",
                                         "val": "base_val.json",
                                         "test": "base_test.json"},
                                "novel": {"1": {"train": ["1_1_train.json", "1_2_train.json", "...", "1_#SAMPLING_train.json"],
                                                "val": ["1_1_val.json", "1_2_val.json", "...", "1_#SAMPLING_val.json"],
                                                "test": ["1_1_test.json", "1_2_test.json", "...", "1_#SAMPLING_test.json"]},
                                          "2": {"train": ["2_1_train.json", "2_2_train.json", "...", "2_#SAMPLING_train.json"],
                                                "val": ["2_1_val.json", "2_2_val.json", "...", "2_#SAMPLING_val.json"],
                                                "test": ["2_1_test.json", "2_2_test.json", "...", "2_#SAMPLING_test.json"]},
                                          "5":{},
                                          "10":{},}
                                }