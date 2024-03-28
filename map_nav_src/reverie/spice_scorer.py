import os,sys
from numpy.lib.npyio import save
import pandas as pd
import ast

from reverie.bleu_coco.bleu import Bleu

class BleuScorer():
    def __init__(self):
        print('setting up scorers...')
        self.scorer = Bleu()
        self.method = "Bleu"

    def prepare_data(self,data):
        """ pre-process the data
        :param data: the list of save_data
        each item includes keys: path_id, Inference(list), Ground Truth(list)
        """
        reference = {}
        ground_truth = {}
        for idx, item in enumerate(data):
            reference[idx] = item['Inference']
            ground_truth[idx] = item['Ground Truth']
        return reference, ground_truth
        
    def compute_scores(self,data):
        reference, ground_truth = self.prepare_data(data)
        score, scores = self.scorer.compute_score(ground_truth,reference)
        return score