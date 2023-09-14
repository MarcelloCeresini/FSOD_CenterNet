import json
import os
import random
import time
from typing import Dict, List
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
        self.idx_to_img = {i: coco.loadImgs(ids=[img])[0]
                           for i, img in enumerate(self.coco.imgs)}

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
        current_image_id = self.idx_to_img[idx]['id']

        img_name = os.path.join(self.images_dir,
                                self.idx_to_img[idx]['file_name'])

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
            if 'iscrowd' in a: del a['iscrowd'] # Pointless to keep it

        sample = {'image': image, 
                  "landmarks": annotations_for_image}

        if isinstance(self.transform, (TransformTraining, TransformTesting)):
            return self.transform(sample)       # sample, transformed_landmarks, original_sample
        else:
            return sample


class DatasetsGenerator():

    def __init__(self, 
            annotations_path: str,
            images_dir: str, 
            novel_class_ids_path: str,
            train_set_path: str,
            val_set_path: str,
            test_set_path: str,
            use_fixed_novel_sets: bool = False,
            novel_train_set_path: str | None = None,
            novel_val_set_path: str | None = None,
            novel_test_set_path: str | None = None,
    ) -> None:
        
        self.images_dir         = images_dir

        self.annotations_path   = annotations_path
        self.train_set_path     = train_set_path
        self.val_set_path       = val_set_path
        self.test_set_path      = test_set_path

        self.train_base         = COCO(train_set_path)
        self.val_base           = COCO(val_set_path)
        self.test_base          = COCO(test_set_path)

        self.use_fixed_novel_sets   = use_fixed_novel_sets
        self.novel_train_set_path   = novel_train_set_path
        self.novel_val_set_path     = novel_val_set_path
        self.novel_test_set_path    = novel_test_set_path

        self.novel_class_ids_path   = novel_class_ids_path

        # Check if we should select a new split of the set or use the given ones
        if self.use_fixed_novel_sets:
            assert self.novel_train_set_path is not None, 'Novel train set path must not be None in fixed set mode'
            assert self.novel_val_set_path is not None, 'Novel val set path must not be None in fixed set mode'
            assert self.novel_test_set_path is not None, 'Novel test set path must not be None in fixed set mode'
            
            # No further checks are done, so make sure that the annotation files are ok
            self.train_novel    = COCO(self.novel_train_set_path)
            self.val_novel      = COCO(self.novel_val_set_path)
            self.test_novel     = COCO(self.novel_test_set_path)
        
        else:
            self.setup()


    def setup(self):
        # Open the full annotations file
        with open(self.annotations_path, 'r') as f:
            self.full_annotations = json.load(f)
        # Create a COCO object to access its API
        self.coco_dset = COCO(self.annotations_path)

        # Load the IDs of the pre-selected novel classes
        with open(self.novel_class_ids_path, 'r') as f:
            self.novel_classes = json.load(f)['novel_cat_ids']


    def _sample_novel_set(self, 
                          sampleable_novel_classes: List[int], 
                          K: int,
                          default_set_path: str,
                          do_not_sample: Dict[int, List[int]] = None) -> COCO:
        # Use our algorithm to collect exactly K annotations for each of the sampleable novel classes
        
        try:
            annots = self.create_annotation_sets_with_K_shots(
                self.coco_dset, sampleable_novel_classes, K = K, 
                do_not_sample = do_not_sample, timeout = 10
            )
        except TimeoutError as e:
            print(e)
            # Fallback to default set
            if os.path.exists(default_set_path):
                return COCO(default_set_path)
            else:
                raise FileNotFoundError('Default novel set not found.')
            
        novel_set = self.convert_k_shot_algo_to_coco(annots, self.full_annotations)
        with TemporaryFile('w+') as fp:
            json.dump(novel_set, fp)
            return COCO(fp.name), annots


    def _generate_new_novel_sets(self, 
                                 K: int, val_K: int, test_K: int,
                                 num_novel_classes_to_sample: int | None = None, 
                                 novel_classes_to_sample_list: List | None = None,
                                 random_seed: int | None = None):
        
        if not num_novel_classes_to_sample and not novel_classes_to_sample_list:
            print("\n\nYou did not specify a number of novel classes to sample nor "
                  "a list of novel classes to allow: this means that all novel classes "
                  "will be used: be aware of that!\n\n")

        self.random_gen = random.Random(random_seed)
        # Starting from here, all calls made to the random generator will be deterministic
        # as long as they will be the same number of calls 

        if not self.use_fixed_novel_sets:

            # Check if we don't want to use all novel classes
            if novel_classes_to_sample_list is not None:
                # Here we want to use exactly the defined novel classes
                sampleable_novel_classes = novel_classes_to_sample_list
            elif num_novel_classes_to_sample is not None:
                # Here we want to sample exactly num_novel_classes
                assert num_novel_classes_to_sample > 0, f'We should sample more than {num_novel_classes_to_sample} base classes.'
                sampleable_novel_classes = random.sample(self.novel_classes, k=num_novel_classes_to_sample)
            else:
                # Here we want to use all the novel classes
                sampleable_novel_classes = self.novel_classes

            # Sample the novel classes
            self.train_novel, train_samples = self._sample_novel_set(sampleable_novel_classes, K, 
                                                      self.novel_train_set_path, 
                                                      do_not_sample = None)
            self.val_novel, val_samples     = self._sample_novel_set(sampleable_novel_classes, val_K,
                                                      self.novel_val_set_path,
                                                      do_not_sample = train_samples)
            self.test_novel, test_samples   = self._sample_novel_set(sampleable_novel_classes, test_K,
                                                      self.novel_test_set_path,
                                                      do_not_sample = {k: train_samples[k] + val_samples[k]
                                                                       for k in train_samples})


    def get_base_sets(self):
        # TODO: validation should have TransformTraining or TransformTesting?
        # ANSWER: Training because it needs labels to compute the loss
        return (
            DatasetFromCocoAnnotations(self.train_base, self.images_dir, TransformTraining(
                base_classes=list(self.train_base.cats),
                novel_classes=list(self.train_novel.cats) if hasattr(self, 'train_novel') else []
            )),
            DatasetFromCocoAnnotations(self.val_base, self.images_dir, TransformTraining(
                base_classes=list(self.val_base.cats),
                novel_classes=list(self.val_novel.cats) if hasattr(self, 'val_novel') else []
            )),
            DatasetFromCocoAnnotations(self.test_base, self.images_dir, TransformTesting(
                base_classes=list(self.test_base.cats),
                novel_classes=list(self.test_novel.cats) if hasattr(self, 'test_novel') else []
            ))
        )
    
    def get_base_sets_dataloaders(self, batch_size = None, num_workers = 1, pin_memory = True,
                                  drop_last = False, shuffle = True):
        train_base, val_base, test_base = self.get_base_sets()
        return (
            DataLoader(dataset=train_base, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle),
            DataLoader(dataset=val_base, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle),
            DataLoader(dataset=test_base, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle)
        )

    def generate_datasets(self, K: int, val_K: int, test_K: int,
                            num_novel_classes_to_sample: int | None = None, 
                            novel_classes_to_sample_list: List | None = None,
                            random_seed: int | None = None):
        '''
        Returns two tuples: one containing train, val, test splits for the base set and the other containing
        train, val, test splits for the novel set. Each split is a `DatasetFromCocoAnnotations` object.
        '''
        self._generate_new_novel_sets(K, val_K, test_K, num_novel_classes_to_sample, 
                                      novel_classes_to_sample_list, random_seed)
        return self.get_base_sets(), (
                DatasetFromCocoAnnotations(self.train_novel, self.images_dir, TransformTraining(
                base_classes=list(self.train_base.cats),
                novel_classes=list(self.train_novel.cats) if hasattr(self, 'train_novel') else novel_classes_to_sample_list
            )),
                DatasetFromCocoAnnotations(self.val_novel, self.images_dir, TransformTraining(
                base_classes=list(self.val_base.cats),
                novel_classes=list(self.val_novel.cats) if hasattr(self, 'val_novel') else novel_classes_to_sample_list
            )),
                DatasetFromCocoAnnotations(self.test_novel, self.images_dir, TransformTesting(
                base_classes=list(self.test_base.cats),
                novel_classes=list(self.test_novel.cats) if hasattr(self, 'test_novel') else novel_classes_to_sample_list
            ))
        )
    

    def generate_dataloaders(self, 
                             K: int, 
                             val_K: int, 
                             test_K: int,
                             num_novel_classes_to_sample: int | None = None, 
                             novel_classes_to_sample_list: List | None = None,
                             gen_random_seed: int | None = None,
                             batch_size: int = None, 
                             num_workers: int = 0,
                             pin_memory: bool = False, 
                             drop_last: bool = False, 
                             shuffle: bool = False):
        '''
        Returns two tuples: one containing train, val, test splits for the base set and the other containing
        train, val, test splits for the novel set. Each split is a `DataLoader` object.
        '''
        (train_base, val_base, test_base), (train_novel, val_novel, test_novel) = \
            self.generate_datasets(K, val_K, test_K, num_novel_classes_to_sample, 
                                   novel_classes_to_sample_list, random_seed=gen_random_seed)
        return  (
                DataLoader(dataset=train_base, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle),
                DataLoader(dataset=val_base, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle),
                DataLoader(dataset=test_base, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle)
            ), (
                DataLoader(dataset=train_novel, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle),
                DataLoader(dataset=val_novel, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle),
                DataLoader(dataset=test_novel, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=drop_last, shuffle=shuffle)
            )

    
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
