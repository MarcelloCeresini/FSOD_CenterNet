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

## Virtual Environment

Install pyenv and pyenv-virtualenv:
Install python 3.11.4:

```shell
    pyenv install 3.11.4
```

Move to repo_folder_path and:

```shell
    pyenv local 3.11.4
    pyenv virtualenv 3.11.4 env-FSOD_CenterNet
    pyenv activate env-FSOD_CenterNet
    python3.11 -m pip install --upgrade pip
    pip install -r requirements.txt
```

