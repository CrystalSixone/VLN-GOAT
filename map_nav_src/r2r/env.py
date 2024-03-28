''' Batched REVERIE navigation environment '''

import json
import os
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict
import copy

import MatterSim
import random

from utils.data import load_nav_graphs, new_simulator
from utils.data import angle_feature, get_all_point_angle_feature, get_view_rel_angles
from utils.data import calculate_vp_rel_pos_fts, get_angle_fts

from r2r.eval_utils import cal_dtw, cal_cls

ERROR_MARGIN = 3.0
MAX_DIST = 30   # normalize
MAX_STEP = 10   # normalize
TRAIN_MAX_STEP = 20

class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, args, connectivity_dir, scan_data_dir=None, feat_db=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.args = args
        self.aug_feat_db = None
        if type(feat_db) is list:
            self.aug_feat_db = feat_db
        else:
            self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            if scan_data_dir:
                sim.setDatasetPath(scan_data_dir)
            sim.setNavGraphPath(connectivity_dir)
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setBatchSize(1)
            sim.initialize()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]
            if self.aug_feat_db is not None:
                if i % 2 == 0:
                    # Alternatively use original features and augmented features
                    feature = self.aug_feat_db[0].get_image_feature(state.scanId, state.location.viewpointId, self.args.img_type)
                else:
                    feature = self.aug_feat_db[1].get_image_feature(state.scanId, state.location.viewpointId, self.args.img_type)
            else:
                feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId, self.args.img_type)
                
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])


class R2RNavBatch(object):
    ''' Implements the REVERIE navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
        self, view_db, instr_data, connectivity_dir, 
        batch_size=64, angle_feat_size=4, seed=0, name=None, sel_data_idxs=None,
        speaker_angle_feat_size=128,tok=None,args=None, scanvp_cands_file=None, save_instr=False
        
    ):
        self.env = EnvBatch(args, connectivity_dir, feat_db=view_db, batch_size=batch_size)
        self.args = args
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.name = name

        self.tok = tok

        self.gt_trajs = self._get_gt_trajs(self.data, save_instr=save_instr) # for evaluation

        # For CFP traj feature extraction
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles]
        if scanvp_cands_file is not None:
            self.scanvp_cands = json.load(open(scanvp_cands_file))

        # in validation, we would split the data
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits 
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx: end_idx]

        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self._load_nav_graphs()

        self.sim = new_simulator(self.connectivity_dir)
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)

        # For speaker.
        if args.use_transpeaker:
            self.speaker_angle_feat_size = speaker_angle_feat_size
            self.speaker_angle_feature = get_all_point_angle_feature(self.sim,self.speaker_angle_feat_size)
        
        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def _get_gt_trajs(self, data, save_instr=False):
        if save_instr:
            gt_trajs = {
                x['instr_id']: (x['scan'], x['path'], x['instruction']) \
                    for x in data if len(x['path']) > 1
            }
        else:
            gt_trajs = {
                x['instr_id']: (x['scan'], x['path']) \
                    for x in data if len(x['path']) > 1
            }
        return gt_trajs

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0
    
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

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)
                    
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'position': (loc.x, loc.y, loc.z),
                        }
                        if self.args.use_transpeaker:
                            speaker_angle_feat = angle_feature(loc_heading, loc_elevation, self.speaker_angle_feat_size)
                            adj_dict[loc.viewpointId]['speaker_feature'] = np.concatenate((visual_feat, speaker_angle_feat), -1)

            candidate = list(adj_dict.values())

            self.buffered_state_dict[long_id] = [
                {key: c[key]
                for key in
                    ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId',
                    'pointId', 'idx', 'position']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                visual_feat = feature[ix]
                c_new['heading'] = c_new['normalized_heading'] - base_heading
                c_new['elevation'] = c_new['normalized_elevation'] - base_elevation
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)

                if self.args.use_transpeaker:
                    speaker_angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.speaker_angle_feat_size)
                    c_new['speaker_feature'] = np.concatenate((visual_feat, speaker_angle_feat), -1)
                    
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            agent_feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : agent_feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instruction' : item['instruction'],
                'instr_encoding': item['instr_encoding'],
                'gt_path' : item['path'],
                'path_id' : item['path_id']
            }
            # for speaker
            if self.args.use_transpeaker:
                speaker_feature = np.concatenate((feature,self.speaker_angle_feature[base_view_id]),-1)
                ob['speaker_feature'] = speaker_feature
                ob['teacher'] = self._teacher_path_action(state, item['path'])
            # RL reward. The negative distance between the state and the final state
            # There are multiple gt end viewpoints on REVERIE. 
            if ob['instr_id'] in self.gt_trajs:
                ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            else:
                ob['distance'] = 0
            
            obs.append(ob)
        return obs
    
    def _teacher_path_action(self, state, path, t=None, shortest_teacher=False):
        if shortest_teacher:
            return self._shortest_path_action(state, path[-1])

        teacher_vp = None
        if t is not None:
            teacher_vp = path[t + 1] if t < len(path) - 1 else state.location.viewpointId
        else:
            if state.location.viewpointId in path:
                cur_idx = path.index(state.location.viewpointId)
                if cur_idx == len(path) - 1: # STOP
                    teacher_vp = state.location.viewpointId
                else:
                    teacher_vp = path[cur_idx + 1]
        return teacher_vp

    def reset(self, batch=None, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is not None:
            self.batch = batch
        else:
            self._next_minibatch(**kwargs)
        
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
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
        gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # shape=(num_gmap_vpids, 7)
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)
        
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i+1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists


    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)
                
        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len+1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts

        return vp_pos_fts
    
    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, scan, pred_path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }
        return avg_metrics, metrics
        
