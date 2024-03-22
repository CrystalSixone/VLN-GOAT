''' This is to build the backdoor dictionaries for images and instructions.
'''
import os,sys
root_path = os.getcwd()
current_path = os.path.join(root_path,'map_nav_src')
sys.path.append(root_path)
sys.path.append(current_path)

import numpy as np
from collections import defaultdict
import base64
import csv
import h5py
import json
import jsonlines
from transformers import BertModel, AutoTokenizer
import ast
from tqdm import tqdm

from utils.data import PickSpecificWords

ROOMTYPE_TSV_FIELDNAMES = ['key', 'room_type']
IMG_TSV_FIELDNAMES = ['roomtype','feature','pz']
INSTR_TSV_FIELDNAMES = ['token_type','token','feature','pz']
INSTR_ALL_FEATURE_FIELDNAMES = ['token_type','token','feature']
TIM_TSV_FIELDNAMES = ['path_id', 'txt_feats', 'vp_feats', 'gmap_feats']
action_list = [
            'right','left','down','up','forward','around','straight',
            'into','front','behind','exit','enter','besides','through',
            'stop','out','wait','passed','climb','leave','past','before','after',
            'between','in','along','cross','end','head','inside','outside','across',
            'towards','face','ahead','toward'
        ]
    
def read_img_tsv(tsv_file):
    in_data = []
    with open(tsv_file, 'rt') as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = IMG_TSV_FIELDNAMES)
        for item in reader:
            item['feature'] = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
            item['pz'] = float(item['pz'])
            in_data.append(item)
    print(1)

def read_instr_tsv(tsv_file):
    in_data = []
    with open(tsv_file, 'rt') as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = INSTR_TSV_FIELDNAMES)
        for item in reader:
            item['feature'] = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
            item['pz'] = float(item['pz'])
            in_data.append(item)
    return in_data

def readUnseenScan(unseen_json):
    ''' read from unseen_enc.json to remove unseen environment
    '''
    unseen_list = []
    with open(unseen_json, 'r') as f:
        data = json.load(f)
    
    for item in data:
        if item['scan'] not in unseen_list:
            unseen_list.append(item['scan'])
    
    return unseen_list

class ImageReader():
    def __init__(self, img_file, roomtype_file, unseen_json, img_feat_size=768, remove_unseen=True):
        self.img_file = img_file
        self.roomtype_file = roomtype_file
        self.img_feat_size = img_feat_size
        self.img_data = {}
        self.roomtype_data = {}

        self.unseen_scan_list = readUnseenScan(unseen_json)
        self.remove_unseen = remove_unseen

        self.get_roomtype_file_from_tsv()
        self.get_feat_from_h5py()

    
    def get_feat_from_h5py(self):
        with h5py.File(self.img_file, 'r') as f:
            for key in f:
                scan = key.split('_')[0]
                if self.remove_unseen and scan in self.unseen_scan_list:
                    continue
                ft = f[key][...][:,:self.img_feat_size].astype(np.float32)
                self.img_data[key] = ft
        print('Load image features finished.')
    
    def get_roomtype_file_from_tsv(self):
        with open(self.roomtype_file, 'r') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = ROOMTYPE_TSV_FIELDNAMES)
            for item in reader:
                scan = item['key'].split('_')[0]
                if self.remove_unseen and scan in self.unseen_scan_list:
                    continue
                data = item['room_type'].replace('[','').replace(']','')
                data_list = data.split(',')
                for i,data_item in enumerate(data_list):
                    data_list[i] = data_item[1:-1]
                    if data_list[i][0] == ' ' or data_list[i][0] == '\'':
                        data_list[i] = data_list[i][1:]
                self.roomtype_data[item['key']] = data_list
        print('Load room type finished.')
    
    def stastic_roomtype(self,roomnum=50):
        total_roomtypes = defaultdict(lambda:0)
        for key, value in self.roomtype_data.items():
            for roomtype in value:
                total_roomtypes[roomtype] += 1
        sorted_roomtype_set = sorted(total_roomtypes.items(),key=lambda x:x[1], reverse=True)[:roomnum]
        print(sorted_roomtype_set)
        return sorted_roomtype_set

    def build_zdict_and_pz(self, roomnum=50, output_file='image_z_dict.tsv'):
        sorted_roomtype_set = self.stastic_roomtype(roomnum)
        roomtype_keys = [item[0] for item in sorted_roomtype_set]

        # pz
        total_number = 0
        pz_dict = {}
        for item in sorted_roomtype_set:
            total_number += item[1]
        for (room, value) in sorted_roomtype_set:
            pz_dict[room] = value/total_number

        total_feature_dict = defaultdict(lambda:[])
        for key, value in self.roomtype_data.items():
            for index_view, room in enumerate(value):
                if room in roomtype_keys:
                    total_feature_dict[room].append(self.img_data[key][index_view])
        
        # Save as tsv file
        output_file = output_file.split('.')[0]+f'_{roomnum}.tsv'
        with open(output_file, 'wt') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = IMG_TSV_FIELDNAMES)
            for room, feature in total_feature_dict.items():
                feature = np.mean(np.array(feature),axis=0)
                record = {
                    'feature': str(base64.b64encode(feature), "utf-8"),
                    'roomtype': room,
                    'pz': pz_dict[room]
                }
                writer.writerow(record)
        print(f'Finished. The tsv file has been saved in {output_file}.')        

class TextReader():
    def __init__(self, json_file, cat_file=None):
        self.json_file = json_file
        self.cat_file = cat_file
        self.instr_data = []

        self.specific_picker = PickSpecificWords(cat_file=cat_file)
        
    def readInstructions(self,dataset='r2r'):
        suffix = self.json_file.split('.')[-1] 
        if 'jsonl' in suffix: 
            with jsonlines.open(self.json_file, 'r') as f:
                for item in f:
                    if dataset == 'rxr':
                        if 'en' not in item['language']:
                            continue
                    if dataset == 'soon':
                        for instr in item['instructions']:
                            self.instr_data.append(instr['full'])
                    else:
                        self.instr_data.append(item['instruction'])
        else:
            with open(self.json_file, 'r') as f:
                self.train_data = json.load(f)
            for item in self.train_data:
                for instr in item['instructions']:
                    self.instr_data.append(instr)
        print('Load %.2f instructions finished.'%(len(self.instr_data)))

    def initModel(self,model='bert'):
        if model == 'bert':
            model_name = "bert-base-uncased"
        elif model == 'roberta':
            model_name = "datasets/pretrained/roberta"
        print(f'model_name:{model_name}')
        self.bert_model = BertModel.from_pretrained(model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model.eval()
        for para in self.bert_model.parameters():
            para.requires_grad = False
    
    def bertEmbedding(self,sentence):
        encoded_input = self.tokenizer(sentence, return_tensors='pt')
        output = self.bert_model(encoded_input)
        return output

    def build_zdict_and_pz(self, model='bert', output_file='instr_z_dict.tsv', dataset='r2r'):
        '''
        model: bert / roberta
        '''
        self.initModel(model=model)
        self.readInstructions(dataset)
        landmark_dict = defaultdict(lambda:[])
        direction_dict = defaultdict(lambda:[])
        # extract instr features
        for instr in tqdm(self.instr_data, desc="Extracting features"):
            landmark_list, direction_list, tokens = self.specific_picker.pick_action_object_words_with_index(instr,map=False)
            input = self.tokenizer.encode_plus(instr,
                                                truncation=True,
                                                return_tensors='pt',
                                                add_special_tokens=True)
            text_index = input.input_ids[:,:250].cuda()
            # This records some words that are splitted into multiple tokens
            re_text = self.tokenizer.decode(text_index.squeeze(),skip_special_tokens=True).split()
            output = self.bert_model(text_index).last_hidden_state.squeeze().cpu().numpy() # [len_token,768]
            # Pick the targeted token embedding
            count = 0
            landmark_idx = 0
            direction_idx = 0
            for i, re_token in enumerate(re_text):
                if re_token[0] == '#':
                    continue
                else:
                    if landmark_idx < len(landmark_list) and count == landmark_list[landmark_idx][0]:
                        key = landmark_list[landmark_idx][1]
                        emb = output[i+1] # output +1 since there is [CLS] token
                        landmark_dict[key].append(emb)
                        landmark_idx += 1
                    if 'reverie' not in output_file and direction_idx < len(direction_list) and count == direction_list[direction_idx][0]:
                        key = direction_list[direction_idx][1]
                        emb = output[i+1]
                        direction_dict[key].append(emb)
                        direction_idx += 1
                    count += 1

        # pz
        landmark_pz_dict = {}
        direction_pz_dict = {}
        l_total_num, d_total_num = 0, 0
        for key, value in landmark_dict.items():
            l_total_num += len(value)
        for key, value in landmark_dict.items():
            landmark_pz_dict[key] = len(value) / l_total_num
        for key, value in direction_dict.items():
            d_total_num += len(value)
        for key, value in direction_dict.items():
            direction_pz_dict[key] = len(value) / d_total_num

        with open(output_file, 'wt') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = INSTR_TSV_FIELDNAMES)
            for key, value in landmark_dict.items():
                feature = np.mean(np.array(value),axis=0)
                record = {
                    'feature': str(base64.b64encode(feature), "utf-8"),
                    'token_type': 'landmark',
                    'pz': landmark_pz_dict[key],
                    'token': key
                }
                writer.writerow(record)
            if 'reverie' not in output_file:
                for key, value in direction_dict.items():
                    feature = np.mean(np.array(value),axis=0)
                    record = {
                        'feature': str(base64.b64encode(feature), "utf-8"),
                        'token_type': 'direction',
                        'pz': direction_pz_dict[key],
                        'token': key
                    }
                    writer.writerow(record)
        print(f'Finished. The tsv file has been saved in {output_file}.')        

    def record_all_token_features(self, output_file='instr_all_features.tsv'):
        self.initModel()
        self.readInstructions()
        landmark_dict = defaultdict(lambda:[])
        direction_dict = defaultdict(lambda:[])
        # extract instr features
        for instr in self.instr_data:
            landmark_list, direction_list, tokens = self.specific_picker.pick_action_object_words_with_index(instr,map=False)
            input = self.tokenizer.encode_plus(instr,
                                                truncation=True,
                                                return_tensors='pt',
                                                add_special_tokens=True)
            text_index = input.input_ids.cuda()
            # This records some words that are splitted into mutliple tokens
            re_text = self.tokenizer.convert_ids_to_tokens(text_index.squeeze(),skip_special_tokens=True)
            output = self.bert_model(text_index).last_hidden_state.squeeze().cpu().numpy() # [len_token,768]
            # Pick the targeted token embedding
            count = 0
            landmark_idx = 0
            direction_idx = 0
            for i, re_token in enumerate(re_text):
                if re_token[0] == '#':
                    continue
                else:
                    if landmark_idx < len(landmark_list) and count == landmark_list[landmark_idx][0]:
                        key = landmark_list[landmark_idx][1]
                        emb = output[i+1] # output +1 since there is [CLS] token
                        landmark_dict[key].append(emb)
                        landmark_idx += 1
                    if direction_idx < len(direction_list) and count == direction_list[direction_idx][0]:
                        key = direction_list[direction_idx][1]
                        emb = output[i+1]
                        direction_dict[key].append(output[i+1])
                        direction_idx += 1
                    count += 1

        with open(output_file, 'wt') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = INSTR_ALL_FEATURE_FIELDNAMES)
            for key, value in landmark_dict.items():
                for feature in value:
                    record = {
                        'feature': str(base64.b64encode(feature), "utf-8"),
                        'token_type': 'landmark',
                        'token': key
                    }
                    writer.writerow(record)
            for key, value in direction_dict.items():
                for feature in value:
                    record = {
                        'feature': str(base64.b64encode(feature), "utf-8"),
                        'token_type': 'direction',
                        'token': key
                    }
                    writer.writerow(record)
        print(f'Finished. The tsv file has been saved in {output_file}.')

if __name__ == '__main__':
    '''1: Load Image room type'''
    print('Start to extract img features')
    roomtype_file = 'datasets/R2R/features/pano_roomtypes.tsv'
    img_file = 'datasets/R2R/features/CLIP-ViT-B-16-views.hdf5'
    unseen_json = 'datasets/R2R/annotations/R2R_val_unseen_enc.json' # to remove unseen scans
    imageReader = ImageReader(img_file, roomtype_file, unseen_json=unseen_json, remove_unseen=True)
    data = imageReader.build_zdict_and_pz(output_file='image_z_dict_clip.tsv')

    '''2: Load Instruction'''
    print('Start to extract txt features')
    model='roberta'
    dataset = 'r2r' # rxr / soon /reverie
    r2r_train_file = 'datasets/R2R/annotations/R2R_train_enc.json'
    # reverie_train_file = 'datasets/REVERIE/annotations/REVERIE_train_enc.json'
    # rxr_train_file = 'datasets/R2R/annotations/RxR/rxr_train_guide.jsonl'
    # soon_train_file = 'datasets/SOON/annotations/bert_enc/train_enc_pseudo_obj_label.jsonl'
    cat_file = 'datasets/R2R/annotations/category_mapping.tsv'
    textReader = TextReader(json_file=r2r_train_file, cat_file=cat_file)
    textReader.build_zdict_and_pz(model=model, output_file=f'{dataset}_z_instr_dict.tsv', dataset=dataset)