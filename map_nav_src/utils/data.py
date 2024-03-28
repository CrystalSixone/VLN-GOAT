import os, sys
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import random
import re
import string
from collections import defaultdict
import spacy
import nltk
from transformers import BertTokenizer
import csv
import base64
from sklearnex import patch_sklearn
patch_sklearn(['KMeans','DBSCAN'])

from sklearn.cluster import KMeans
import joblib
import time


class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, scan, viewpoint, type='hdf5'):
        if type == 'hdf5':
            return self.get_image_feature_from_h5py(scan, viewpoint)
        elif type == 'tsv':
            return self.get_image_feature_from_tsv(scan, viewpoint)

    def get_image_feature_from_h5py(self, scan, viewpoint):
        # read image features from .h5py
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft
    
    def get_image_feature_from_tsv(self, scan, viewpoint):
        # read image features from .tsv
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
            ft = self._feature_store[key]
        return ft
    
def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()

    return sim

def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)

def get_point_angle_feature(sim, angle_feat_size, baseViewId=0):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature

def get_all_point_angle_feature(sim, angle_feat_size):
    return [get_point_angle_feature(sim, angle_feat_size, baseViewId) for baseViewId in range(36)]

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz/xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts

def get_view_rel_angles(baseViewId=0):
    rel_angles = np.zeros((36, 2), dtype=np.float32)

    base_heading = (baseViewId % 12) * math.radians(30)
    base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    for ix in range(36):
        if ix == 0:
            heading = 0
            elevation = math.radians(-30)
        elif ix % 12 == 0:
            heading = 0
            elevation += math.radians(30)
        else:
            heading += math.radians(30)
        rel_angles[ix, 0] = heading - base_heading
        rel_angles[ix, 1] = elevation - base_elevation

    return rel_angles

class PickSpecificWords():
    def __init__(self, cat_file=None):
        self.anno_path = 'datasets/R2R/annotations/R2R_%s_enc.json'
        self.spacy_model = spacy.load("en_core_web_sm")
        self.action_list = [
            'right','left','down','up','forward','around','straight',
            'into','front','behind','exit','enter','besides','through',
            'stop','out','wait','passed','climb','leave','past','before','after',
            'between','in','along','cross','end','head','inside','outside','across',
            'towards','face','ahead','toward'
        ]
        self.cat_file = cat_file
        if self.cat_file is not None:
            self.cat_mapping, self.category_number = self.read_category_file(self.cat_file)
            self.lem = nltk.stem.wordnet.WordNetLemmatizer()
            self.action_map = {}
            for index, val in enumerate(self.action_list):
                self.action_map[val] = index
    
    def read_category_file(self,infile):
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
    
    def pick_action_object_words(self,instr,map=True):
        tokens = self.spacy_model(instr)
        action_list = []
        object_list = []
        # record_list: record the word should be masked.
        # mask_id_list: record the index of the word in bert tokens.
        for num,token in enumerate(tokens):
            if token.pos_ == 'NOUN':
                # focus on NOUN
                name = re.sub(r'[^\w\s]',' ',str(token).lower().strip())
                name = self.lem.lemmatize(name) # convert the word into root word
                name = ''.join([i for i in name if not i.isdigit()]) # remove number
                if name in self.cat_mapping.keys():
                    name_map = self.cat_mapping[name]
                    if name_map in self.category_number.keys():
                        if map:
                            object_list.append(self.category_number[name_map]+1) 
                        else:
                            object_list.append(name_map)
            if str(token).lower() in self.action_list:
                # focus on ACTION
                if map:
                    action_list.append(self.action_map[str(token).lower()]+1)
                else:
                    action_list.append(str(token).lower())
        return object_list, action_list

    def pick_action_object_words_with_index(self,instr,map=True):
        tokens = self.spacy_model(instr)
        action_list = []
        object_list = []
        # record_list: record the word should be masked.
        # mask_id_list: record the index of the word in bert tokens.
        for num,token in enumerate(tokens):
            if token.pos_ == 'NOUN':
                # focus on NOUN
                name = re.sub(r'[^\w\s]',' ',str(token).lower().strip())
                name = self.lem.lemmatize(name) # convert the word into root word
                name = ''.join([i for i in name if not i.isdigit()]) # remove number
                if name in self.cat_mapping.keys():
                    name_map = self.cat_mapping[name]
                    if name_map in self.category_number.keys():
                        if map:
                            object_list.append((num,self.category_number[name_map]+1)) 
                        else:
                            object_list.append((num,name_map))
            if str(token).lower() in self.action_list:
                # focus on ACTION
                if map:
                    action_list.append((num, self.action_map[str(token).lower()]+1)) 
                else:
                    action_list.append((num,str(token).lower()))
        return object_list, action_list, tokens

class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character

    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        self.index_to_word = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
            new_w2i = defaultdict(lambda: self.word_to_index['<UNK>'])
            new_w2i.update(self.word_to_index)
            self.word_to_index = new_w2i
            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key
        old = self.vocab_size()
        self.add_word('<BOS>')
        assert self.vocab_size() == old+1

    def finalize(self):
        """
        This is used for debug
        """
        self.word_to_index = dict(self.word_to_index)   

    def add_word(self, word):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()    
        self.index_to_word[self.vocab_size()] = word

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def vocab_size(self):
        return len(self.index_to_word)

    def encode_sentence(self, sentence, max_length=None):
        if max_length is None:
            max_length = self.encoding_length
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = [self.word_to_index['<BOS>']]
        for word in self.split_sentence(sentence):
            if word in self.word_to_index.keys():
                encoding.append(self.word_to_index[word])   # Default Dict
            else:
                encoding.append(self.word_to_index['<UNK>'])
        encoding.append(self.word_to_index['<EOS>'])

        if len(encoding) <= 2:
            return None
        #assert len(encoding) > 2

        if len(encoding) < max_length:
            encoding += [self.word_to_index['<PAD>']] * (max_length-len(encoding))  # Padding
        elif len(encoding) > max_length:
            encoding[max_length - 1] = self.word_to_index['<EOS>']                  # Cut the length with EOS

        return np.array(encoding[:max_length])

    def decode_sentence(self, encoding, length=None):
        sentence = []
        if length is not None:
            encoding = encoding[:length]
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.index_to_word[ix])
        return " ".join(sentence)

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
        
        new_inst = []
        for i in range(len(inst)-1):
            if inst[i] == inst[i+1]:
                continue
            new_inst.append(inst[i])
        inst = new_inst

        end = np.argmax(np.array(inst) == self.word_to_index['<EOS>'])     # If no <EOS>, return empty string
        if end == 0: # 没有<eos>
            end = len(inst)
            
        if len(inst) > 1 and inst[0] == self.word_to_index['<BOS>']:
            start = 1
        else:
            start = 0
        # print(inst, start, end)
        return inst[start: end]
    
# ==========
# Use KMeans to randomly pick features
# ==========
class KMeansPicker():
    def __init__(self, front_feat_file, kmeans_file=None, n_clusters=256):
        self.TIM_TSV_FIELDNAMES = ['path_id', 'txt_feats', 'vp_feats', 'gmap_feats']
        self.n_clusters = n_clusters
        txt_feats, vp_feats, gmap_feats = self.read_tim_tsv(front_feat_file)
        self.feat_dicts = {
            'txt_feats': txt_feats,
            'vp_feats': vp_feats,
            'gmap_feats': gmap_feats
        }
        if kmeans_file is not None:
            self.kmeans_model_dict = {
                'txt_feats': joblib.load(os.path.join(kmeans_file,'txt_feats.pkl')),
                'vp_feats': joblib.load(os.path.join(kmeans_file,'vp_feats.pkl')),
                'gmap_feats': joblib.load(os.path.join(kmeans_file,'gmap_feats.pkl')),
            }
        else:
            self.kmeans_model_dict = {}
            for k, v in self.feat_dicts.items():
                kmeans = KMeans(n_clusters=n_clusters)
                # start_time = time.time()
                kmeans.fit(v)
                # print('Finish KMeans on %d %s. The total time is %.2f min.' %(n_clusters, k, (time.time()-start_time)/60))
                self.kmeans_model_dict[k] = kmeans

        # print('Successfully load front-back features.')

    def read_tim_tsv(self, front_feat_file, return_dict=False):
        txt_feats = []
        vp_feats = []
        gmap_feats = []
        with open(front_feat_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.TIM_TSV_FIELDNAMES)
            for item in reader:
                txt_feats.append(np.frombuffer(base64.b64decode(item['txt_feats']), dtype=np.float32))
                vp_feats.append(np.frombuffer(base64.b64decode(item['vp_feats']), dtype=np.float32))
                gmap_feats.append(np.frombuffer(base64.b64decode(item['gmap_feats']), dtype=np.float32))
        if return_dict:
            feat_dict = {
                'txt_feats': txt_feats,
                'vp_feats': vp_feats,
                'gmap_feats': gmap_feats
            }
            return feat_dict
        else:
            return np.array(txt_feats), np.array(vp_feats), np.array(gmap_feats)

    def random_pick_front_features(self):
        random_feat_dicts = defaultdict(lambda: [])
        for k in self.feat_dicts.keys():
            kmeans = self.kmeans_model_dict[k]
            # Iterate over the unique cluster labels
            for cluster_label in np.unique(kmeans.labels_):
                # Find the indices of samples belonging to the current cluster
                cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]
                
                # Randomly select one sample from the current cluster
                random_sample_index = np.random.choice(cluster_indices)
                random_sample = self.feat_dicts[k][random_sample_index]
                
                # Add the randomly picked sample to the list
                random_feat_dicts[k].append(random_sample)

        return random_feat_dicts

    def save_features(self, args, feat_data):
        # Save frontdoor features for inference
        target_file = os.path.join(args.z_front_log_dir, f'frontdoor_update_features.tsv')
        with open(target_file, 'wt') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = self.TIM_TSV_FIELDNAMES)
            for i in range(self.n_clusters):
                record = {
                    'path_id': 0,
                    'txt_feats': str(base64.b64encode(feat_data['txt_feats'][i]), "utf-8"),
                    'vp_feats': str(base64.b64encode(feat_data['vp_feats'][i]), "utf-8"),
                    'gmap_feats': str(base64.b64encode(feat_data['gmap_feats'][i]), "utf-8"),
                }
                writer.writerow(record)