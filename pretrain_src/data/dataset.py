'''
Instruction and trajectory dataset
'''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import jsonlines
import numpy as np
import h5py
import math
import re
import nltk
import lmdb
import base64
import pickle
import csv
import torch
import random
import time

from .common import load_nav_graphs
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax

from utils.logger import LOGGER

MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20

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

def preprocess_name(name,cat_mapping,cat_number):
    ''' preprocess the name of object
    '''
    name = re.sub(r'[^\w\s]',' ',str(name).lower().strip())
    lem = nltk.stem.wordnet.WordNetLemmatizer()
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
    def __init__(self, img_zdict_file, obj_zdict_file, txt_zdict_file):
        self.img_tsv_fieldnames = ['roomtype','feature','pz']
        self.txt_tsv_fieldnames = ['token_type','token','feature','pz']
        self.img_zdict_file = img_zdict_file
        self.obj_zdict_file = obj_zdict_file
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
                img_features.append(np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32))
                img_pzs.append(float(item['pz']))
        return {
            "img_features": torch.from_numpy(np.array(img_features)),
            "img_pzs": torch.from_numpy(np.array(img_pzs))
        }

    def load_instr_tensor(self):
        instr_direction_features = []
        instr_direction_pzs = []
        instr_landmark_features = []
        instr_landmark_pzs = []
        with open(self.txt_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.txt_tsv_fieldnames)
            for item in reader:
                if item['token_type'] == 'direction':
                    instr_direction_features.append(np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32))
                    instr_direction_pzs.append(float(item['pz']))
                elif item['token_type'] == 'landmark':
                    instr_landmark_features.append(np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32))
                    instr_landmark_pzs.append(float(item['pz']))
        return {
            "instr_direction_features": torch.from_numpy(np.array(instr_direction_features)),
            "instr_direction_pzs": torch.from_numpy(np.array(instr_direction_pzs)),
            "instr_landmark_features": torch.from_numpy(np.array(instr_landmark_features)),
            "instr_landmark_pzs": torch.from_numpy(np.array(instr_landmark_pzs)),
        }

class ReverieTextPathData(object):
    def __init__(
        self, anno_files, img_ft_db, obj_ft_db, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        cat_file=None,args=None,tok=None,
        aug_img_db=None, z_dicts=None
    ):  
        self.args = args
        self._feature_store = img_ft_db
        self._obj_feat_store = obj_ft_db
        
        self.cat_file = cat_file
        self.object_data = None
        self.tok = tok
        if self.cat_file is not None:
            self.cat_mapping, self.category_number = read_category_file(self.cat_file)
        else:
            self.cat_mapping, self.category_number = None, None
            
        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640
        self.obj_image_size = 480 * 640

        self.max_txt_len = max_txt_len
        self.max_objects = max_objects
        self.act_visited_node = act_visited_node

        self.in_memory = in_memory

        # {scan_vp: {vp: [viewidx, rel_angle_dist, rel_heading, rel_elevation]}}
        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles]

        self.data = []
        for anno_file in anno_files:
            with jsonlines.open(anno_file, 'r') as f:
                data = list(f)
                for i, item in enumerate(data):
                    if 'language' in item.keys():
                        # For RxR
                        if 'en' not in item['language']:
                            continue
                        new_item = {}
                        instr = item['instruction']
                        new_item['instruction'] = instr
                        new_item['instr_encoding'] = self.tok(instr,padding=True,truncation=True,max_length=self.args.max_instr_len)['input_ids']
                        new_item['path_id'] = item['path_id']
                        new_item['heading'] = item['heading']
                        new_item['instr_id'] = item['instruction_id']
                        new_item['scan'] = item['scan']
                        new_item['path'] = item['path']
                        self.data.append(new_item)
                    else:
                        item['instr'] = []
                        if 'pos_vps' in item.keys(): # reverie
                            item['objId'] = item['instr_id'].split('_')[1]
                        self.data.append(item)
                        
                    if self.args.debug and i >= 50:
                        break
        
        if aug_img_db is not None:
            # self.aug_img_file = aug_img_db
            self.use_aug_ft = True
            self._aug_feature_store = aug_img_db
            print("Using augmented image feature in EnvEdit!")
        else:
            self.use_aug_ft = False
        
        self.z_dicts = z_dicts

    def __len__(self):
        return len(self.data)

    def get_scanvp_feature(self, scan, viewpoint, type='tsv'):
        key = str(scan) + '_' + str(viewpoint)
        if self._obj_feat_store is not None:
            obj_attrs = {}
            obj_ft = np.zeros((0, self.obj_feat_size+self.obj_prob_size), dtype=np.float32)
            if key in self._obj_feat_store:
                obj_ft, obj_attrs = self._obj_feat_store[key]
                
        if self.use_aug_ft: 
            if np.random.rand() > 0.5:
                img_ft = self.get_aug_image_feature(scan, viewpoint)
            else:
                if type == 'hdf5':
                    img_ft = self.get_image_feature_from_h5py(scan, viewpoint)
                elif type == 'tsv':
                    img_ft = self.get_image_feature_from_tsv(scan, viewpoint)
        else:
            if type == 'hdf5':
                    img_ft = self.get_image_feature_from_h5py(scan, viewpoint)
            elif type == 'tsv':
                img_ft = self.get_image_feature_from_tsv(scan, viewpoint)
        
        return img_ft, obj_ft, obj_attrs

    def get_image_feature_from_h5py(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft
    
    def get_image_feature_from_tsv(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        views = 36
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            scanIds = []
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
            scanIds_viewpointsId = {}
            with open(self.img_ft_file, "r") as tsv_in_file:     # Open the tsv file.
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
                for item in reader:
                    scanId = item['scanId']
                    if scanId not in scanIds:
                        scanIds.append(scanId)
                        scanIds_viewpointsId[scanId] = []
                        scanIds_viewpointsId[scanId].append(item['viewpointId'])
                    else:
                        scanIds_viewpointsId[scanId].append(item['viewpointId'])
                    long_id = item['scanId'] + "_" + item['viewpointId']
                    ft = np.frombuffer(base64.decodebytes(item['features'].encode('ascii')),
                                                    dtype=np.float32).reshape((views, -1))
                    self._feature_store[long_id] = ft
        return ft

    def get_aug_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        views = 36
        if key in self._aug_feature_store:
            ft = self._aug_feature_store[key]
        else:
            scanIds = []
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
            scanIds_viewpointsId = {}

            with open(self.aug_img_file, "r") as tsv_in_file:     # Open the tsv file.
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
                for item in reader:
                    scanId = item['scanId']
                    if scanId not in scanIds:
                        scanIds.append(scanId)
                        scanIds_viewpointsId[scanId] = []
                        scanIds_viewpointsId[scanId].append(item['viewpointId'])
                    else:
                        scanIds_viewpointsId[scanId].append(item['viewpointId'])
                    long_id = item['scanId'] + "_" + item['viewpointId']
                    ft = np.frombuffer(base64.decodebytes(item['features'].encode('ascii')),
                                                    dtype=np.float32).reshape((views, -1))
                    self._feature_store[long_id] = ft
        return ft

    def get_obj_label(self, item, last_vp_objids):
        if 'objId' in item.keys():
            gt_obj_id = item['objId'] # Aug
        else:
            gt_obj_id = item['instr_id'].split('_')[1] # By default, instr_id in reverie pre-train contains pathId_objId_instrId
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            # it occurs when the gt_objid is not in max_objects
            obj_label = -100 # ignore 
            # print('No groundtruth obj_id', item['instr_id'], len(obj_ids))
        return obj_label

    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                        + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k # [stop] is 0
            # local: 
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1 # [stop] is 0
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        pos_vps = item['pos_vps']
        gt_path = item['path']

        if end_vp is None:
            if end_vp_type == 'pos': # pos
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
            elif end_vp_type == 'neg_in_gt_path':
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]

        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
        
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_reverie_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids, traj_reverie_obj_names = self.get_traj_pano_fts(scan, gt_path, cur_heading, cur_elevation)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr': item['instr'][:self.max_txt_len],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_reverie_loc_fts': traj_reverie_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            'vp_angles': last_vp_angles,
            'traj_reverie_obj_names': traj_reverie_obj_names
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)
    

        return outs

    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            heading = (viewidx % 12) * math.radians(30)
            elevation = (viewidx // 12 - 1) * math.radians(30)
        return heading, elevation

    def get_traj_pano_fts(self, scan, path, cur_heading, cur_elevation):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []
        traj_reverie_loc_fts = []
        traj_reverie_obj_names = []
        for vp in path:
            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_angle = self.all_point_rel_angles[12][v[0]]
                if self.args.correct_heading: 
                    heading = cur_heading - view_angle[0] + v[2]
                    elevation = cur_elevation - view_angle[1] + v[3]
                else:
                    heading = view_angle[0] + v[2]
                    elevation = view_angle[1] + v[3]
                view_angles.append([heading, elevation])
                cand_vpids.append(k)
                
            # non cand views
            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)

            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            obj_names = np.array([0],dtype=np.int)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h/self.obj_image_h, w/self.obj_image_w, (h*w)/self.obj_image_size]           
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)
                obj_names = np.array(obj_attrs['names']).astype('int')

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            traj_reverie_obj_names.append(obj_names)
            traj_reverie_loc_fts.append(np.concatenate([obj_ang_fts, obj_box_fts], 1))
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_reverie_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids, traj_reverie_obj_names
        
    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]

        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s'%(scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        if self.act_visited_node: # R2R default False
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # shape=(num_gmap_vpids, 7)
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)
        
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i+1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists
    
    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'], 
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                    (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
        
    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)
                
        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len+1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts

        return vp_pos_fts

class R2RTextPathData(ReverieTextPathData):
    def __init__(
        self, anno_files, img_ft_db, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        cat_file=None,args=None,tok=None,
        aug_img_db=None, z_dicts=None
    ):
        super().__init__(
            anno_files, img_ft_db, None, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=0, obj_prob_size=0, 
            max_objects=0, max_txt_len=max_txt_len, in_memory=in_memory,
            act_visited_node=act_visited_node,
            cat_file=cat_file,args=args,tok=tok,
            aug_img_db=aug_img_db, z_dicts=z_dicts
        )

    def get_scanvp_feature(self, scan, viewpoint, type='hdf5'):
        if self.use_aug_ft: 
            if np.random.rand() > 0.5:
                return self.get_aug_image_feature(scan, viewpoint)
            else:
                if type == 'hdf5':
                    return self.get_image_feature_from_h5py(scan, viewpoint)
                elif type == 'tsv':
                    return self.get_image_feature_from_tsv(scan, viewpoint)
        else:
            if type == 'hdf5':
                    return self.get_image_feature_from_h5py(scan, viewpoint)
            elif type == 'tsv':
                return self.get_image_feature_from_tsv(scan, viewpoint)


    def get_act_labels(self, end_vp, end_idx, item, gmap_vpids, traj_cand_vpids):
        if end_vp == item['path'][-1]:  # stop
            global_act_label = local_act_label = 0
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            gt_next_vp = item['path'][end_idx + 1]
            for k, cand_vp in enumerate(gmap_vpids):
                if cand_vp == gt_next_vp:
                    global_act_label = k
                    break
            # local: 
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                if cand_vp == gt_next_vp:
                    local_act_label = k + 1 # [stop] is 0
                    break
        return global_act_label, local_act_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, end_vp=None
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item['heading']
        gt_path = item['path']

        if end_vp is None: # None for R2R
            if end_vp_type == 'pos': # True endpoint
                # name convention with REVERIE (last vp)
                end_idx = len(gt_path) - 1
                end_vp = gt_path[-1]
            elif end_vp_type in ['neg_in_gt_path', 'neg_others']: # Negative endpoint
                # name convention with REVERIE (mid vps in the path)
                end_vps = gt_path[:-1]
                end_idx = np.random.randint(len(end_vps))
                end_vp = end_vps[end_idx]
        else:
            assert end_vp in gt_path
            end_idx = gt_path.index(end_vp)
            
        gt_path = gt_path[:end_idx+1]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
        
        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles = self.get_traj_pano_fts(scan, gt_path, cur_heading, cur_elevation)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)

        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_reverie_loc_fts': None,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,

            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,

            'vp_pos_fts': vp_pos_fts,
            'vp_angles': last_vp_angles,
        }
        if 'instr' in item.keys():
            outs['instr'] = item['instr'][:self.max_txt_len]

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, end_idx, item, gmap_vpids, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label

        if return_img_probs:
            # TODO: whether adding gmap img probs
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
        
        if self.z_dicts is not None:
            img_zdict = self.z_dicts['img_zdict']
            instr_zdict = self.z_dicts['instr_zdict']

            outs['instr_z_direction_features'] = instr_zdict['instr_direction_features']
            outs['instr_z_direction_pzs'] = instr_zdict['instr_direction_pzs']
            outs['instr_z_landmark_features'] = instr_zdict['instr_landmark_features']
            outs['instr_z_landmark_pzs'] = instr_zdict['instr_landmark_pzs']

            outs['img_z_features'] = img_zdict['img_features']
            outs['img_z_pzs'] = img_zdict['img_pzs']

        return outs

    def get_traj_pano_fts(self, scan, path, cur_heading, cur_elevation):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], []

        for vp in path:
            view_fts = self.get_scanvp_feature(scan, vp)
            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0])
                view_img_fts.append(view_fts[v[0]])
                view_angle = self.all_point_rel_angles[12][v[0]]
                if self.args.correct_heading: 
                    heading = cur_heading - view_angle[0] + v[2]
                    elevation = cur_elevation - view_angle[1] + v[3]
                else:
                    heading = view_angle[0] + v[2]
                    elevation = view_angle[1] + v[3]
                view_angles.append([heading, elevation])
                cand_vpids.append(k)

            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
    
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_loc_fts.append(np.concatenate([view_ang_fts, view_box_fts], 1))
            traj_nav_types.append([1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)))

            traj_cand_vpids.append(cand_vpids)
            last_vp_angles = view_angles

        return traj_view_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, last_vp_angles


class SoonTextPathData(ReverieTextPathData):
    def __init__(
        self, anno_files, img_ft_db, obj_ft_db, scanvp_cands_file, connectivity_dir,
        image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        cat_file=None, args=None, tok=None,
        aug_img_db=None
    ):
        super().__init__(
            anno_files, img_ft_db, obj_ft_db, scanvp_cands_file, connectivity_dir,
            image_feat_size=image_feat_size, image_prob_size=image_prob_size,
            angle_feat_size=angle_feat_size, obj_feat_size=obj_feat_size, 
            obj_prob_size=obj_prob_size, max_objects=max_objects, 
            max_txt_len=max_txt_len, in_memory=in_memory,
            act_visited_node=act_visited_node,
            cat_file=cat_file,args=args,tok=tok,
            aug_img_db=aug_img_db
        )
        self.obj_image_h = self.obj_image_w = 600
        self.obj_image_size = 600 * 600

    def get_obj_label(self, item, last_vp_objids):
        obj_label = item['obj_pseudo_label']['idx']
        if obj_label >= self.max_objects:
            obj_label = -100
        return obj_label

    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, end_vp=None
    ):
        if end_vp_type == 'pos':
            end_vp = self.data[idx]['path'][-1]
        return super().get_input(
            idx, end_vp_type, 
            return_img_probs=return_img_probs, 
            return_act_label=return_act_label, 
            return_obj_label=return_obj_label, 
            end_vp=end_vp
        )

def read_img_features_from_h5py(ft_file, img_ft_size=768):
    feature_store = {}
    with h5py.File(ft_file, 'r') as f:
        for key in f.keys():
            # ft = f[key][...][:, :img_ft_size].astype(np.float32)
            ft = f[key][...][:, :].astype(np.float32)
            feature_store[key] = ft
    return feature_store

def read_img_features_from_tsv(ft_file):
    print("Start loading the image feature ...")
    start = time.time()
    views = 36
    tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
    features = {}
    with open(ft_file, "r") as tsv_in_file:     # Open the tsv file.
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
        for item in reader:
            long_id = item['scanId'] + "_" + item['viewpointId']
            features[long_id] = np.frombuffer(base64.decodebytes(item['features'].encode('ascii')),
                                                dtype=np.float32).reshape((views, -1))   
    print("Finish Loading the image feature from %s in %0.4f seconds" % (ft_file, time.time() - start))
    return features
    
def read_reverie_obj_features(obj_file, max_objects, obj_feat_size, obj_prob_size,cat_mapping,category_number):
    print("Start loading the object feature ...")
    feature_store = {}
    with h5py.File(obj_file, 'r') as f:
        for key in f.keys():
            obj_attrs = {}
            obj_fts = np.zeros((0, obj_feat_size+obj_prob_size), dtype=np.float32)
            obj_fts = f[key][...].astype(np.float32)
            obj_fts = obj_fts[:max_objects]
            for attr_key, attr_value in f[key].attrs.items():
                if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids', 'names']:
                    if attr_key == 'names':
                        for i in range(len(attr_value)):
                            _, attr_value[i] = preprocess_name(attr_value[i],cat_mapping,category_number)
                    obj_attrs[attr_key] = attr_value[:max_objects]
                    
            feature_store[key] = [obj_fts, obj_attrs]
    return feature_store

def read_soon_obj_features(obj_file, max_objects, obj_feat_size, obj_prob_size):
    print("Start loading the object feature ...")
    feature_store = {}
    with h5py.File(obj_file, 'r') as f:
        for key in f.keys():
            obj_attrs = {}
            obj_fts = np.zeros((0, obj_feat_size+obj_prob_size), dtype=np.float32)
            with h5py.File(obj_file, 'r') as f:
                if key in f:
                    obj_fts = f[key][...].astype(np.float32)
                    obj_fts = obj_fts[:max_objects]
                    for attr_key, attr_value in f[key].attrs.items():
                        if attr_key in ['directions', 'bboxes', 'obj_ids']:
                            obj_attrs[attr_key] = attr_value[:max_objects]
                    obj_attrs['bboxes'] = np.array(obj_attrs['bboxes']).astype(np.float32) # 4
                    obj_attrs['sizes'] = np.zeros((len(obj_attrs['bboxes']), 2), dtype=np.float32)
                    obj_attrs['sizes'][:, 0] = obj_attrs['bboxes'][:, 2] - obj_attrs['bboxes'][:, 0]
                    obj_attrs['sizes'][:, 1] = obj_attrs['bboxes'][:, 3] - obj_attrs['bboxes'][:, 1]
            feature_store[key] = [obj_fts, obj_attrs]
    return feature_store