from fashionpedia.fp import Fashionpedia
from PIL import Image, ImageOps
import os
import torch
from torch.utils.data import Dataset

class SketchyDataset(Dataset):

    def __init__(self, dataset_root, split='train', 
                 load_img=False, 
                 load_global_sketch=False,
                 load_local_sketch=False,
                 img_size=512,
                 img_transforms=None,
                 global_sketch_transforms=None,
                 local_sketch_transforms=None,
                 text_tokenizers = None,
                 with_shoes=False, # shoes are not included by default
                 concat_locals=True, # concatenate local descriptions to create the global description
                 compose_global_sketch=True, # compose the global sketch from the local sketches instead of using the pre-computed one
                 ):
        self.root = dataset_root
        self.split = split
        self.load_img = load_img
        self.load_global_sketch = load_global_sketch
        self.load_local_sketch = load_local_sketch
        self.img_size = img_size
        self.img_transforms = img_transforms
        self.global_sketch_transforms = global_sketch_transforms
        self.local_sketch_transforms = local_sketch_transforms
        self.text_tokenizers = text_tokenizers
        self.concat_locals = concat_locals
        self.with_shoes = with_shoes
        self.compose_global_sketch = compose_global_sketch
        
        if self.compose_global_sketch:
            assert load_global_sketch and load_local_sketch, "Need to load both global and local sketches to compose the global sketch"
        
        self.json_path = os.path.join(self.root, f"{self.split}_sketchy.json")

        self.init_dataset(self.json_path)
    
    def init_dataset(self, json_path):
        self.fp = Fashionpedia(json_path)
        # go through the dataset and remove the shoes
        if not self.with_shoes:
            self.removeShoes()
        # get all images ids
        self.img_ids = list(self.fp.getImgIds())

    def collate_fn(self, batch):
        """ Use this when you are ok with having lists of different sizes in the batch"""
        return_dict = {}
        for key in batch[0].keys():
            if key == 'image':
                images = [d[key] for d in batch]
                if self.img_transforms is not None:
                    images = torch.stack(images)
                return_dict['image'] = images
            else:
                return_dict[key] = [d[key] for d in batch]
        return return_dict
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        return_dict = {}
        img_id = self.img_ids[idx]
        return_dict['image_id'] = img_id
        img_data = self.fp.loadImgs(img_id)[0]
        return_dict['img_data'] = img_data
        img_path = os.path.join(self.root, self.split, 'images', str(img_id), img_data['file_name'])
        global_sketch_path = os.path.join(self.root, self.split, 'full_sketches', str(img_id), str(img_id) + '.png')
        annotations = self.fp.loadAnns(self.fp.getAnnIds(img_id))
        return_dict['annotations'] = annotations
        return_dict['global_sketch_path'] = global_sketch_path

        return_dict['local_descriptions'] = [ann['description'].strip().lower() for ann in annotations]
        return_dict['local_descriptions_ann_ids'] = [ann['id'] for ann in annotations]
        if self.concat_locals:
            return_dict['global_description'] = ". ".join(return_dict['local_descriptions']).strip().lower()
            
        return_dict['local_sketches_paths'] = [os.path.join(self.root, self.split, 'partial_sketches', str(img_id), str(ann['id']) + '.png') for ann in annotations]
        if self.load_local_sketch:
            local_sketches = [Image.open(local_sketch_path).resize((self.img_size, self.img_size)) for local_sketch_path in return_dict['local_sketches_paths']]
            if self.compose_global_sketch:

                global_sketch = Image.new("L", (self.img_size, self.img_size), color=0)
                for local_sketch in local_sketches:
                    local_sketch = ImageOps.invert(local_sketch.convert("L"))
                    global_sketch.paste(local_sketch, (0, 0), local_sketch)
                global_sketch = ImageOps.invert(global_sketch)
                global_sketch = global_sketch.convert("RGB")
                if self.global_sketch_transforms is not None:
                    global_sketch = self.global_sketch_transforms(global_sketch)
                return_dict['global_sketch'] = global_sketch
            local_sketches = [local_sketch.convert("RGB") for local_sketch in local_sketches]
            if self.local_sketch_transforms is not None:
                local_sketches = [self.local_sketch_transforms(local_sketch) for local_sketch in local_sketches]
            return_dict['local_sketches'] = local_sketches
        else:
            return_dict['local_sketches'] = return_dict['local_sketches_paths']
        return_dict['image_path'] = img_path
        if self.load_img:
            image = Image.open(img_path).convert("RGB")
            image = image.resize((self.img_size, self.img_size))
            if self.img_transforms is not None:
                image = self.img_transforms(image)
            return_dict['image'] = image
        else:
            return_dict['image'] = img_path

        if not self.compose_global_sketch:
            if self.load_global_sketch:
                global_sketch = Image.open(global_sketch_path).convert("RGB")
                global_sketch = global_sketch.resize((self.img_size, self.img_size))
                if self.global_sketch_transforms is not None:
                    global_sketch = self.global_sketch_transforms(global_sketch)
                return_dict['global_sketch'] = global_sketch
            else:
                return_dict['global_sketch'] = global_sketch_path

        # process text with tokenizers if needed
        if self.text_tokenizers is not None:
            # first global description
            text = return_dict['global_description']
            if len(self.text_tokenizers) == 1:
                text_input_ids = self.text_tokenizers[0](
                    text,
                    max_length=self.text_tokenizers[0].model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                return_dict['global_description_ids'] = text_input_ids

                # then local descriptions
                local_descriptions = return_dict['local_descriptions']
                local_text_ids = []
                for text in local_descriptions:
                    text_input_ids = self.text_tokenizers[0](
                        text,
                        max_length=self.text_tokenizers[0].model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids
                    local_text_ids.append(text_input_ids)
                return_dict['local_descriptions_ids'] = local_text_ids
            else:
                # get text and tokenize
                text_input_ids = self.text_tokenizers[0](
                    text,
                    max_length=self.text_tokenizers[0].model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                
                text_input_ids_2 =self.text_tokenizers[1](
                    text,
                    max_length=self.text_tokenizers[1].model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                return_dict['global_description_ids'] = text_input_ids
                return_dict['global_description_ids_2'] = text_input_ids_2

                # then local descriptions
                local_descriptions = return_dict['local_descriptions']
                local_text_ids = []
                for text in local_descriptions:
                    text_input_ids = self.text_tokenizers[0](
                        text,
                        max_length=self.text_tokenizers[0].model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids
                    local_text_ids.append(text_input_ids)
                return_dict['local_descriptions_ids'] = local_text_ids
                local_text_ids_2 = []
                for text in local_descriptions:
                    text_input_ids_2 = self.text_tokenizers[1](
                        text,
                        max_length=self.text_tokenizers[1].model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids
                    local_text_ids_2.append(text_input_ids_2)
                return_dict['local_descriptions_ids_2'] = local_text_ids_2
        return return_dict
    
    def ann2Mask(self, ann):
        mask = self.fp.annToMask(ann)*255
        mask = Image.fromarray(mask)
        mask = ImageOps.contain(mask, (ann['final_width'], ann['final_height']))
        padding = tuple(ann['padding'])
        mask = ImageOps.expand(mask, padding, fill='black')
        mask = mask.resize((self.img_size, self.img_size))
        return mask
    
    def removeShoes(self):
        # get the annotations from the fp object
        new_annotations = []
        for ann_id, ann in self.fp.anns.items():
            # remove all annotations with category_name "shoe"
            if ann['category_name'] != 'shoe':
                new_annotations.append(ann.copy())
        self.fp.dataset['annotations'] = new_annotations
        # re-create the index
        self.fp.createIndex()
        # get all images ids
        self.img_ids = list(self.fp.getImgIds())
        # remove images that have no annotations
        new_img_data = []
        for img_id, img_data in self.fp.imgs.items():
            anns = self.fp.loadAnns(self.fp.getAnnIds(img_id))
            if len(anns) > 0:
                new_img_data.append(img_data.copy())
        self.fp.dataset['images'] = new_img_data
        # re-create the index
        self.fp.createIndex()