import os
import sys
root_path = os.getcwd()
current_path = os.path.join(root_path,'map_nav_src')
sys.path.append(root_path)
sys.path.append(current_path)

import MatterSim

import argparse
import numpy as np
import json
import math
from PIL import Image
import csv
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import AutoProcessor, BlipForQuestionAnswering

TSV_FIELDNAMES = ['key','room_type']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60

def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids

def build_feature_extractor(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_path)
    model = BlipForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=model_path)
    model.to(device)
    model.eval()
    for para in model.parameters():
        para.requires_grad = False

    return model, processor, device

def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def process_features(proc_id, out_queue, scanvp_list, args):
    print('start proc_id: %d' % proc_id)
    prompt = 'What kind of room is this?'

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, processor, device = build_feature_extractor(args.model_path)
    for scan_id, viewpoint_id in scanvp_list:
        # Loop all discretized views from this location
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = Image.fromarray(image[:, :, ::-1]) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image.save('{}_{}.jpg'.format(scan_id,ix))
            images.append(image)

        # images = torch.stack([image for image in images], 0)
        generated_room_types = []
        for k in range(0, len(images), args.batch_size):
            inputs = processor(images=images[k: k+args.batch_size],text=prompt,return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            generated_text = processor.decode(outputs[0],skip_special_tokens=True)
            generated_room_types.append(generated_text)

        out_queue.put((scan_id, viewpoint_id, generated_room_types))

    out_queue.put(None)


def build_feature_file(args):
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()

    with open(args.output_file, 'w') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = TSV_FIELDNAMES)
        while num_finished_workers < num_workers:
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, data = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                record = {
                    'key': key,
                    'room_type': data
                }
                writer.writerow(record)
                num_finished_vps += 1
                progress_bar.update(num_finished_vps)
    
    progress_bar.finish()
    for process in processes:
        process.join()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../blip-vqa-base')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='../connectivity')
    # parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--scan_dir',default='../vln/v1/v1/scans')
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--output_file')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    build_feature_file(args)


