# FSOD_CenterNet

Implementation of Few-Shot Object Detection through a CenterNet architecture

## Dataset

Taken from [FSOD GitHub Repo](https://github.com/fanq15/Few-Shot-Object-Detection-Dataset)
Download dataset and annotation from: [FSOD Dataset](https://drive.google.com/drive/folders/1XXADD7GvW8M_xzgFpHfudYDYtKtDgZGM)

Insert them in your data structure as follows:

```nothing
  MAIN_FOLDER_PATH
      └── repo_folder
            ├── src
            ├── ...
            │ 
            └── data
                  ├──── fsod
                          ├── annotations
                          │       ├── fsod_train.json
                          │       └── fsod_test.json
                          └── images
                                ├── part_1
                                └── part_2
```

