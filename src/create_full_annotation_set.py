import json
import os
from typing import Any, Dict, Optional

# Code taken from https://gradiant.github.io/pyodi/reference/apps/coco-merge/#pyodi.apps.coco.coco_merge.coco_merge
# It's used to merge a training and a validation coco file into a single larger set.
def coco_merge(
    input_extend: str, input_add: str, output_file: str, indent: Optional[int] = None,
) -> str:
    """Merge COCO annotation files.

    Args:
        input_extend: Path to input file to be extended.
        input_add: Path to input file to be added.
        output_file : Path to output file with merged annotations.
        indent: Argument passed to `json.dump`. See https://docs.python.org/3/library/json.html#json.dump.
    """
    with open(input_extend, "r") as f:
        data_extend = json.load(f)
    with open(input_add, "r") as f:
        data_add = json.load(f)

    output: Dict[str, Any] = {
        k: data_extend[k] for k in data_extend if k not in ("images", "annotations")
    }

    output["images"], output["annotations"] = [], []

    for i, data in enumerate([data_extend, data_add]):

        cat_id_map = {}
        for new_cat in data["categories"]:
            new_id = None
            for output_cat in output["categories"]:
                if new_cat["name"] == output_cat["name"]:
                    new_id = output_cat["id"]
                    break

            if new_id is not None:
                cat_id_map[new_cat["id"]] = new_id
            else:
                new_cat_id = max(c["id"] for c in output["categories"]) + 1
                cat_id_map[new_cat["id"]] = new_cat_id
                new_cat["id"] = new_cat_id
                output["categories"].append(new_cat)

        img_id_map = {}
        for image in data["images"]:
            n_imgs = len(output["images"])
            img_id_map[image["id"]] = n_imgs
            image["id"] = n_imgs

            output["images"].append(image)

        for annotation in data["annotations"]:
            n_anns = len(output["annotations"])
            annotation["id"] = n_anns
            annotation["image_id"] = img_id_map[annotation["image_id"]]
            annotation["category_id"] = cat_id_map[annotation["category_id"]]

            output["annotations"].append(annotation)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=indent)

    return output_file


if __name__ == '__main__':

    base_path = os.path.join('..', '..')
    # Assumes we have train_2017_bboxes.json (https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip)
    # and val_2017_bboxes.json (https://ml-inat-competition-datasets.s3.amazonaws.com/2017/val_2017_bboxes.zip)
    # into the data folder. Creates a full_2017_bboxes.json file in the same folder.
    annotation_file = os.path.join(base_path, 'data', 'train_2017_bboxes.json')
    val_annotation_file = os.path.join(base_path, 'data', 'val_2017_bboxes.json')
    full_annotation_file = os.path.join(base_path, 'data', 'full_2017_bboxes.json')

    # Merge the two JSONs
    coco_merge(annotation_file, val_annotation_file, full_annotation_file)