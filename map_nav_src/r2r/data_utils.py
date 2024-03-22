import os
import json
import jsonlines
import numpy as np
import re
import csv
import base64
import torch

def read_category_file(infile):
    category_mapping = {}
    category_list = []
    category_number = {}
    with open(infile, 'r',encoding='utf-8') as f:
        next(f) 
        for line in f:
            line = line.strip('\n').split('\t')  
            source_name, target_category = line[1], line[-1]
            category_mapping[source_name] = target_category
            if target_category not in category_list:
                category_list.append(target_category)
        category_list.append('others')
        for i,cat in enumerate(category_list):
            category_number[cat] = i
    return category_mapping, category_number

def preprocess_name(name,cat_mapping,cat_number,lem):
    ''' preprocess the name of object
    '''
    name = re.sub(r'[^\w\s]',' ',str(name).lower().strip())
    name = lem.lemmatize(name) # convert the word into root word
    name = ''.join([i for i in name if not i.isdigit()]) # remove number
    if name in cat_mapping:
        name = cat_mapping[name]
    else:
        name = name.split(' ')[0]
        if name in cat_mapping:
            name = cat_mapping[name]
        else:
            name = 'others'
    number = cat_number[name]
    return name, number
   
class LoadZdict():
    def __init__(self, img_zdict_file, txt_zdict_file):
        self.img_tsv_fieldnames = ['roomtype','feature','pz']
        self.txt_tsv_fieldnames = ['token_type','token','feature','pz']
        self.img_zdict_file = img_zdict_file
        self.txt_zdict_file = txt_zdict_file

    def read_img_tsv(self):
        in_data = []
        with open(self.img_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.img_tsv_fieldnames)
            for item in reader:
                item['feature'] = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                item['pz'] = float(item['pz'])
                in_data.append(item)
        return in_data

    def read_instr_tsv(self):
        in_data = []
        with open(self.txt_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.txt_tsv_fieldnames)
            for item in reader:
                item['feature'] = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                item['pz'] = float(item['pz'])
                in_data.append(item)
        return in_data
    
    def load_all_zdicts(self):
        img_zdict, instr_zdict = self.read_img_tsv(), self.read_instr_tsv()
        return img_zdict, instr_zdict
    
    def load_img_tensor(self):
        img_features = []
        img_pzs = []
        with open(self.img_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.img_tsv_fieldnames)
            for item in reader:
                img_feature = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                img_features.append(img_feature)
                img_pzs.append(float(item['pz']))
        return {
            "img_features": torch.from_numpy(np.array(img_features)).cuda(),
            "img_pzs": torch.from_numpy(np.array(img_pzs)).cuda()
        }

    def load_instr_tensor(self, is_random=False):
        instr_direction_features = []
        instr_direction_pzs = []
        instr_landmark_features = []
        instr_landmark_pzs = []
        with open(self.txt_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.txt_tsv_fieldnames)
            for item in reader:
                if item['token_type'] == 'direction':
                    feature = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                    if is_random:
                        instr_direction_features.append(np.random.random(feature.shape).astype(np.float32))
                    else:
                        instr_direction_features.append(feature)
                    instr_direction_pzs.append(float(item['pz']))
                elif item['token_type'] == 'landmark':
                    feature = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                    if is_random:
                        instr_landmark_features.append(np.random.random(feature.shape).astype(np.float32))
                    else:
                        instr_landmark_features.append(feature)
                    instr_landmark_pzs.append(float(item['pz']))
        if len(instr_direction_features) != 0:
            return {
                "instr_direction_features": torch.from_numpy(np.array(instr_direction_features)).cuda(),
                "instr_direction_pzs": torch.from_numpy(np.array(instr_direction_pzs)).cuda(),
                "instr_landmark_features": torch.from_numpy(np.array(instr_landmark_features)).cuda(),
                "instr_landmark_pzs": torch.from_numpy(np.array(instr_landmark_pzs)).cuda(),
            }
        else: # reverie
            return {
                "instr_landmark_features": torch.from_numpy(np.array(instr_landmark_features)).cuda(),
                "instr_landmark_pzs": torch.from_numpy(np.array(instr_landmark_pzs)).cuda(),
            }

def load_instr_datasets(anno_dir, dataset, splits, tokenizer):
    data = []
    for split in splits:
        if "rxr" in dataset:
            # RXR
            filepath = os.path.join(anno_dir, 'RxR', '%s_%s_guide.jsonl' % (dataset,split))
            new_data = []
            with jsonlines.open(filepath, 'r') as f:
                for item in f:
                    new_data.append(item)
        else:
            # R2R
            if "/" not in split:    # the official splits
                if tokenizer == 'bert':
                    filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
                elif tokenizer == 'xlm':
                    filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
                elif tokenizer == 'roberta':
                    filepath = os.path.join(anno_dir, '%s_%s_roberta_enc.json' % (dataset.upper(), split))
                else:
                    raise NotImplementedError('unspported tokenizer %s' % tokenizer)

                with open(filepath) as f:
                    new_data = json.load(f)

                if split == 'val_train_seen':
                    new_data = new_data[:50]

            else:   # augmented data
                print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
                with open(split) as f:
                    new_data = json.load(f)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, for_debug=False, tok=None, is_rxr=False):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer)):
        if is_rxr:
            if 'en' not in item['language']:
                continue # only use english
            new_item = {}
            instr = item['instruction']
            new_item['instruction'] = instr
            new_item['instr_encoding'] = tok(instr,padding=True,truncation=True,max_length=max_instr_len)['input_ids']
            new_item['path_id'] = item['path_id']
            new_item['heading'] = item['heading']
            new_item['instr_id'] = item['instruction_id']
            new_item['scan'] = item['scan']
            new_item['path'] = item['path']
            data.append(new_item)
            if for_debug and i >= 50:
                break # debug
        else:
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                del new_item['instructions']
                del new_item['instr_encodings']
                data.append(new_item)
                if for_debug and i >= 50:
                    break # debug
        
    return data