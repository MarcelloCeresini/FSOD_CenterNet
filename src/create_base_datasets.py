import json
import os
import random
from typing import List

from pycocotools.coco import COCO

BASE_SET_RATIOS = (80, 10, 10)
BASE_SET_SMALL_RATIOS = (8, 1, 1)


def get_train_val_test_annots_and_imgs(coco_dset: COCO, base_classes: List, small:bool=False):
    base_cat_to_imgs = {k: v for k, v in coco_dset.catToImgs.items() if k in base_classes}

    train_set_base_imgs = []
    val_set_base_imgs   = []
    test_set_base_imgs  = []

    for base_cat in base_cat_to_imgs:
        imgs = set(base_cat_to_imgs[base_cat])
        
        n_sample_train  = int((BASE_SET_RATIOS[0] if not small else BASE_SET_SMALL_RATIOS[0]) / 100 * len(imgs))
        n_sample_val    = int((BASE_SET_RATIOS[1] if not small else BASE_SET_SMALL_RATIOS[1]) / 100 * len(imgs))
        n_sample_test   = int((BASE_SET_RATIOS[2] if not small else BASE_SET_SMALL_RATIOS[2]) / 100 * len(imgs))

        train_imgs  = set(random.sample(sorted(imgs), n_sample_train))
        imgs        = imgs.difference(train_imgs)
        val_imgs    = set(random.sample(sorted(imgs), n_sample_val))
        if not small:
            test_imgs = imgs.difference(val_imgs)
        else:
            imgs = imgs.difference(val_imgs)
            test_imgs = set(random.sample(sorted(imgs), n_sample_test))

        train_set_base_imgs.extend(train_imgs)
        val_set_base_imgs.extend(val_imgs)
        test_set_base_imgs.extend(test_imgs)

    print("Base images: \ntrain, val,  set")
    print(len(train_set_base_imgs), len(val_set_base_imgs), len(test_set_base_imgs))

    train_set = coco_dset.loadAnns(coco_dset.getAnnIds(imgIds=train_set_base_imgs))
    val_set   = coco_dset.loadAnns(coco_dset.getAnnIds(imgIds=val_set_base_imgs))
    test_set  = coco_dset.loadAnns(coco_dset.getAnnIds(imgIds=test_set_base_imgs))

    print()
    print("Base annotations: \ntrain, val,  set")
    print(len(train_set), len(val_set), len(test_set))

    return (train_set, val_set, test_set), \
        (coco_dset.loadImgs(train_set_base_imgs), 
         coco_dset.loadImgs(val_set_base_imgs), 
         coco_dset.loadImgs(test_set_base_imgs))


if __name__ == '__main__':
    root_path = os.path.join('..', '..')

    # We assume to have the full dataset and the list of base class ids
    # (those can be regenerated from the notebook in noteboooks/data_tests.ipynb)
    full_annotation_file = os.path.join(root_path, 'data', 'full_2017_bboxes.json')
    base_classes_file = os.path.join(root_path, 'data', 'base_class_ids.json')

    # Create full annotation COCO datasets
    coco_dset = COCO(full_annotation_file)
    
    # Keep the JSON file in memory
    coco_json = json.load(open(full_annotation_file, 'r'))
    # Keep the base classes in memory
    with open(base_classes_file, 'r') as f:
        base_classes = json.load(f)['base_cat_ids']
    # Setup coco format for our datasets
    coco_format = {
        'info': coco_json['info'],
        'licenses': coco_json['licenses'],
        'categories': [c for c in coco_json['categories'] if c['id'] in base_classes],
        'images': [],
        'annotations': []
    }

    # Create the datasets
    print("LARGE DATASET")
    (train_annots, val_annots, test_annots), (train_imgs, val_imgs, test_imgs) = \
        get_train_val_test_annots_and_imgs(coco_dset, base_classes, small=False)

    print('\n\n')
    print("SMALL DATASET")
    (small_train_annots, small_val_annots, small_test_annots), \
        (small_train_imgs, small_val_imgs, small_test_imgs) = \
            get_train_val_test_annots_and_imgs(coco_dset, base_classes, small=True)

    # Save the datasets: first the large sets...
    coco_format['images'] = train_imgs
    coco_format['annotations'] = train_annots
    with open(os.path.join(root_path, 'data', 'base_dset', 'base_train.json'), 'w') as f:
        json.dump(coco_format, f)

    coco_format['images'] = val_imgs
    coco_format['annotations'] = val_annots
    with open(os.path.join(root_path, 'data', 'base_dset', 'base_val.json'), 'w') as f:
        json.dump(coco_format, f)

    coco_format['images'] = test_imgs
    coco_format['annotations'] = test_annots
    with open(os.path.join(root_path, 'data', 'base_dset', 'base_test.json'), 'w') as f:
        json.dump(coco_format, f)

    # Then the small sets
    coco_format['images'] = small_train_imgs
    coco_format['annotations'] = small_train_annots
    with open(os.path.join(root_path, 'data', 'base_dset', 'small_base_train.json'), 'w') as f:
        json.dump(coco_format, f)

    coco_format['images'] = small_val_imgs
    coco_format['annotations'] = small_val_annots
    with open(os.path.join(root_path, 'data', 'base_dset', 'small_base_val.json'), 'w') as f:
        json.dump(coco_format, f)

    coco_format['images'] = small_test_imgs
    coco_format['annotations'] = small_test_annots
    with open(os.path.join(root_path, 'data', 'base_dset', 'small_base_test.json'), 'w') as f:
        json.dump(coco_format, f)