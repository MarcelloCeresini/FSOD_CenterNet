import json
import os
import random
import time
from typing import Any, Dict, List
from tempfile import TemporaryFile

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from pycocotools.coco import COCO

from .transform import TransformTesting, TransformTraining


class DatasetFromCocoAnnotations(Dataset):

    def __init__(self, coco: COCO, images_dir: str, 
                 transform: TransformTesting | TransformTraining) -> None:
        super().__init__()
        self.coco = coco
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.coco.imgs)

    def __getitem__(self, idx):
        '''
        Returns a sample of the dataset. If no transform is passed, the sample is a dictionary with:
            - image
            - landmarks:
                - id
                - category_id
                - center_point
                - size
                - bbox
                - area
                - image_id
            - original_image_size
        '''
        current_image_id = self.coco.imgs[idx]["id"]

        img_name = os.path.join(self.images_dir,
                                self.coco.imgs[idx]["file_name"])

        image = read_image(img_name)

        # Add center point and size to annotations
        annotations_for_image = self.coco.imgToAnns[current_image_id]
        for a in annotations_for_image:
            # a["bbox"] is top [left x position, top left y position, width, height]
            center_point = (a["bbox"][0] + a["bbox"][2]/2, 
                            a["bbox"][1] + a["bbox"][3]/2)
            size = (a["bbox"][2], 
                    a["bbox"][3])
            a['center_point'] = center_point
            a['size'] = size
            del a['iscrowd'] # pointless to keep it

        sample = {'image': image, 
                  "landmarks": annotations_for_image}

        if isinstance(self.transform, (TransformTraining, TransformTesting)):
            sample, transformed_landmarks, original_sample = self.transform(sample)
            return sample, transformed_landmarks, original_sample
        else:
            return sample


class DatasetsGenerator():

    def __init__(self, 
                annotations_path: str,
                images_dir: str, 
                novel_class_ids_path: str,
                train_set_path: str,
                val_set_path: str,
                use_fixed_sets: bool = False,
                K: int = 10,
                val_K: int = 20,
                num_base_classes: int | None = 100,
                num_novel_classes: int | None = 10,
                novel_classes_list: List[int] | None = None,
                novel_train_set_path: str | None = None,
                novel_val_set_path: str | None = None,
                to_be_shown: bool = False
    ) -> None:
        
        self.images_dir = images_dir

        # Check if we should select a new training/validation set or use the given ones
        if use_fixed_sets:
            assert novel_train_set_path is not None, 'Novel train set path must not be None in fixed set mode'
            assert novel_val_set_path is not None, 'Novel val set path must not be None in fixed set mode'
            
            # No further checks are done, so make sure that the annotation files are ok
            self.train_base     = COCO(train_set_path)
            self.val_base       = COCO(val_set_path)
            self.train_novel    = COCO(novel_train_set_path)
            self.val_novel      = COCO(novel_val_set_path)

        else:

            # Open the full annotations file if it exist, otherwise create it
            if not os.path.exists(annotations_path):
                print("Creating full annotations file...")
                self.coco_merge(train_set_path, val_set_path, annotations_path)

            with open(annotations_path, 'r') as f:
                full_annotations = json.load(f)

            # Create a COCO object to access its API
            coco_dset = COCO(annotations_path)

            # Load the IDs of the pre-selected novel classes
            with open(novel_class_ids_path, 'r') as f:
                novel_classes = json.load(f)['novel_cat_ids']

            # Check if we don't want to use all novel classes
            if num_novel_classes is not None:
                # Here we want to sample exactly num_novel_classes
                assert num_novel_classes > 0, f'We should sample more than {num_novel_classes} base classes.'
                sampleable_novel_classes = random.sample(novel_classes, k=num_novel_classes)
            elif novel_classes_list is not None:
                # Here we want to use exactly the defined novel classes
                sampleable_novel_classes = novel_classes_list
            else:
                # Here we want to use all the novel classes
                sampleable_novel_classes = novel_classes

            # Use our algorithm to collect exactly K annotations for each of the sampleable novel classes
            try:
                train_annots = self.create_annotation_sets_with_K_shots(
                    coco_dset, sampleable_novel_classes, K = K, timeout = 10
                )
            except TimeoutError as e:
                print(e)
                # Fallback to default training set
                if os.path.exists(novel_train_set_path):
                    with open(novel_train_set_path, "r") as f:
                        self.train_novel = json.load(f)
                else:
                    raise FileNotFoundError('Default novel train set not found.')
                
            # The validation set should not sample from the images in the training set
            try:
                val_annots = self.create_annotation_sets_with_K_shots(
                    coco_dset, sampleable_novel_classes, K = val_K, 
                    do_not_sample = train_annots, timeout = 10
                )
            except TimeoutError as e:
                print(e)    
                # Fallback to default training set
                if os.path.exists(novel_val_set_path):
                    with open(novel_val_set_path, "r") as f:
                        self.val_novel = json.load(f)
                else:
                    raise FileNotFoundError('Default novel val set not found.')
                
            # Now we should convert the output from the algorithm into COCO format
            novel_train_set = self.convert_k_shot_algo_to_coco(train_annots, full_annotations)
            novel_val_set   = self.convert_k_shot_algo_to_coco(val_annots, full_annotations)
            # Cursing against COCO api because it does not accept COCO-ready dicts but it wants files,
            # we create a temporary file with our dictionary
            with TemporaryFile('w+') as fp:
                json.dump(novel_train_set, fp)
                self.train_novel = COCO(fp.name)
            with TemporaryFile('w+') as fp:
                json.dump(novel_val_set, fp)
                self.val_novel = COCO(fp.name)
            
            # TODO: deal with base classes


    def generate_datasets(self):
        '''
        Returns the train (base and novel) and validation (base and novel) sets for the current run.
        '''
        return  DatasetFromCocoAnnotations(self.train_base, self.images_dir, TransformTraining()), \
                DatasetFromCocoAnnotations(self.train_novel, self.images_dir, TransformTraining()), \
                DatasetFromCocoAnnotations(self.val_base, self.images_dir, TransformTesting()), \
                DatasetFromCocoAnnotations(self.val_novel, self.images_dir, TransformTesting())
    

    def generate_dataloaders(self, batch_size: int = None, num_workers: int = 0,
                             pin_memory: bool = False, drop_last: bool = False, 
                             shuffle: bool = False):
        train_base, train_novel, val_base, val_novel = self.generate_datasets()
        return  DataLoader(dataset=train_base, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle), \
                DataLoader(dataset=train_novel, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle), \
                DataLoader(dataset=val_base, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle), \
                DataLoader(dataset=val_novel, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle)


    def coco_merge(self, 
                   input_extend: str, 
                   input_add: str, 
                   output_file: str, 
                   indent: int | None = None,
    ) -> str:
        """Merge COCO annotation files.
        Source: Code taken from https://gradiant.github.io/pyodi/reference/apps/coco-merge/#pyodi.apps.coco.coco_merge.coco_merge

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

    
    def create_annotation_sets_with_K_shots(
            self, coco_dset: COCO, novel_classes: List[int],
            K: int, do_not_sample: Dict[int, List[int]] | None = None,
            timeout: int = 10) -> Dict[int, Dict]:
        '''
        Samples exactly `K` annotations for each of the `novel_classes` of the input `coco_dset`.
        If a sampled annotation is part of an image that also contains annotations of another novel class,
        the other novel class is populated with annotations from the same image, in order to fully capture
        all annotations in an image. Given the random nature of the algorithm, a timeout can be provided.

        Args:
        - `coco_dset`: The COCO object for the dataset managing annotations and image IDs
        - `noval_classes`: A list of novel class IDs to sample for
        - `K`: The number of annotations to sample for each class
        - `do_not_sample` (Optional, Default None): A dict containing previously-sampled annotations 
            for each class, to avoid sampling them a second time if the algorithm is used more than once 
            (e.g. for a validation or a test set)
        - `timeout` (Optional, Default 10): Number of seconds to let the algorithm run
        '''
        assert K > 0, "K must be positive"
        st_t = time.time()

        # Collect images that should be excluded from sampling
        if do_not_sample is not None:
            do_not_sample_images = {}
            for k in do_not_sample:
                do_not_sample_images[k] = [x['image_id'] for x in do_not_sample[k]]
        
        # Consistency loop
        stop_flag = False
        while not stop_flag:
            
            # Step 1): randomize priority of novel classes
            novel_classes = random.sample(novel_classes, len(novel_classes))
            novel_classes_set = set(novel_classes)
            
            # Create new set of annotations
            annots = {class_id: [] for class_id in novel_classes}
            
            # Step 2): Constructively add images to our annotation pool
            for class_id in tqdm(novel_classes):
                
                # Get the images containing annotations of that class and randomize them
                class_image_ids = coco_dset.getImgIds(catIds=[class_id])
                random.shuffle(class_image_ids)
                
                # Remove from the set of image ids those that we should not sample (if any)
                if do_not_sample is not None:
                    class_image_ids = list(set(class_image_ids) - set(do_not_sample_images[class_id]))
                
                # Start choosing images to fill the class annotations
                for img_id in class_image_ids:
                    
                    # Get annotations for that image
                    img_annots = coco_dset.getAnnIds(imgIds=[img_id])
                    img_annots = coco_dset.loadAnns(img_annots)
                    # Separate annotations into "annotations of that class" and 
                    # "annotations of another class"
                    class_annots = []; non_class_annots = []
                    for ann in img_annots:
                        # Note: we only consider novel classes here. A base class in the same image
                        # is fine.
                        if    ann['category_id'] not in novel_classes_set: continue
                        elif  ann['category_id'] == class_id: class_annots.append(ann)
                        else: non_class_annots.append(ann)
                    
                    # Checks:
                    
                    # 1) The annots could be added to the class list without overflowing K annotations
                    # (otherwise choose another image)
                    if len(annots[class_id]) + len(class_annots) > K:
                        continue
                    
                    # 2) The annotations of other classes in the image don't overflow their respective classes
                    # (otherwise choose another image)
                    ncann_elems = {}
                    for ncann in non_class_annots:
                        if ncann['category_id'] not in ncann_elems:
                            ncann_elems[ncann['category_id']] = 1
                        else:
                            ncann_elems[ncann['category_id']] += 1
                    if len(ncann_elems) > 0 and any([len(annots[k]) + ncann_elems[k] > K for k in ncann_elems]):
                        continue
                    
                    # If all went well, simply add the annotations into their respective classes
                    for ann in img_annots:
                        annots[ann['category_id']].append(ann)
                    
                    # Break the cycle early if we have reached the correct amount of annotations for this class
                    if len(annots[class_id]) == K:
                        break
                
                # Check that the class actually has K annotations and no other class has more than K,
                # otherwise restart from scratch
                if len(annots[class_id]) != K or any([len(annots[x]) > K for x in annots]):
                    break
                # If all classes have exactly K annotations, stop!
                if all([len(annots[x]) == K for x in annots]):
                    stop_flag = True
            
            end_t = time.time()
            if end_t - st_t > timeout:
                raise TimeoutError("Annotation Sampling function was not able to complete sampling in time. Try lowering K.")
            if not stop_flag:
                print("[ANNOTATION SAMPLING] Restarting...")
    
        return annots
    

    def convert_k_shot_algo_to_coco(self, k_shot_anns: Dict, full_anns: Dict):
        k_shot_img_ids = set([img_id['image_id'] for cl_img in k_shot_anns.values() for img_id in cl_img])
        k_shot_ann_ids = set([img_id['id'] for cl_img in k_shot_anns.values() for img_id in cl_img])
        novel_categories = set(k_shot_anns.keys())
        # Copy values from the full annotation set but only for the chosen ids
        coco_anns = {
            'info': full_anns['info'],
            'licenses': full_anns['licenses'],
            'categories': [c for c in full_anns['categories'] if c['id'] in novel_categories],
            'images': [im for im in full_anns['images'] if im['id'] in k_shot_img_ids],
            'annotations': [an for an in full_anns['annotations'] if an['id'] in k_shot_ann_ids]
        }
        return coco_anns
