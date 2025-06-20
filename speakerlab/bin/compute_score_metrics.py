# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import re
import argparse
import numpy as np
from tqdm import tqdm
from kaldiio import ReadHelper
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from speakerlab.utils.utils import get_logger
from speakerlab.utils.score_metrics import (compute_pmiss_pfa_rbst, compute_eer, compute_c_norm)

parser = argparse.ArgumentParser(description='Compute score and metrics')
parser.add_argument('--enrol_data', default='', type=str, help='Enroll data dir')
parser.add_argument('--test_data', default='', type=str, help='Test data dir')
parser.add_argument('--scores_dir', default='', type=str, help='Scores dir')
parser.add_argument('--trials', nargs='+', help='Trial')
parser.add_argument('--p_target', default=0.01, type=float, help='p_target in DCF')
parser.add_argument('--c_miss', default=1, type=float, help='c_miss in DCF')
parser.add_argument('--c_fa', default=1, type=float, help='c_fa in DCF')

def plot_eer_curves(fnr, fpr, scores, thres, labels, save_path):
    plt.figure(figsize=(15, 5))
    
    # Plot 1: FNR vs FPR
    plt.subplot(131)
    plt.plot(fpr, fnr, 'b-', label='ROC')
    plt.plot([0, 1], [0, 1], 'r--', label='EER line')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('FNR vs FPR')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Error Rates vs Scores
    plt.subplot(132)
    # Create threshold points for plotting
    thresholds = np.sort(scores)
    fnrs = []
    fprs = []
    for t in thresholds:
        predictions = (scores >= t).astype(int)
        fn = np.sum((predictions == 0) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        tp = np.sum((predictions == 1) & (labels == 1))
        fnrs.append(fn / (fn + tp) if (fn + tp) > 0 else 1.)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 1.)
    
    plt.plot(thresholds, fnrs, 'b-', label='FNR')
    plt.plot(thresholds, fprs, 'r-', label='FPR')
    plt.axvline(x=thres, color='g', linestyle='--', label=f'EER Threshold')
    plt.xlabel('Score Threshold')
    plt.ylabel('Error Rate')
    plt.title('Error Rates vs Score Threshold')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Score Distribution
    plt.subplot(133)
    plt.hist(scores[labels==1], bins=50, density=True, alpha=0.7, label='Target', color='g')
    plt.hist(scores[labels==0], bins=50, density=True, alpha=0.7, label='Non-target', color='r')
    plt.axvline(x=thres, color='b', linestyle='--', label=f'EER Threshold: {thres:.3f}')
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.scores_dir, exist_ok=True)

    result_path = os.path.join(args.scores_dir, 'result.metrics')
    logger = get_logger(fpath=result_path, fmt = "%(message)s")

    def collect(data_dir):
        data_dict = {}
        emb_arks = [os.path.join(data_dir, i) for i in os.listdir(data_dir) if re.search('.ark$',i)]
        if len(emb_arks) == 0:
            raise Exception(f'No embedding ark files found in {data_dir}')

        # load embedding data
        for ark in emb_arks:
            with ReadHelper(f'ark:{ark}') as reader:
                for key, array in reader:
                    data_dict[key] = array

        return data_dict

    enrol_dict = collect(args.enrol_data)
    test_dict = collect(args.test_data)

    for trial in args.trials:
        scores = []
        labels = []

        trial_name = os.path.basename(trial)
        score_path = os.path.join(args.scores_dir, f'{trial_name}.score')
        with open(trial, 'r') as trial_f, open(score_path, 'w') as score_f:
            lines = trial_f.readlines()
            for line in tqdm(lines, desc=f'scoring trial {trial_name}'):
                pair = line.strip().split()
                enrol_emb, test_emb = enrol_dict[pair[0]], test_dict[pair[1]]
                cosine_score = cosine_similarity(enrol_emb.reshape(1, -1),
                                              test_emb.reshape(1, -1))[0][0]
                # write the score
                score_f.write(' '.join(pair)+' %.5f\n'%cosine_score)
                scores.append(cosine_score)
                if pair[2] == '1' or pair[2] == 'target':
                    labels.append(1)
                elif pair[2] == '0' or pair[2] == 'nontarget':
                    labels.append(0)
                else:
                    raise Exception(f'Unrecognized label in {line}.')

        # compute metrics
        scores = np.array(scores)
        labels = np.array(labels)

        fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
        eer, thres = compute_eer(fnr, fpr, scores)
        min_dcf = compute_c_norm(fnr,
                                fpr,
                                p_target=args.p_target,
                                c_miss=args.c_miss,
                                c_fa=args.c_fa)
                                
        # Plot EER curves
        plot_path = os.path.join(args.scores_dir, f'{trial_name}_eer_curves.png')
        plot_eer_curves(fnr, fpr, scores, thres, labels, plot_path)

        # write the metrics
        logger.info("Results of {} is:".format(trial_name))
        logger.info("EER = {0:.4f}".format(100 * eer))
        logger.info("EER_thres = {0:.4f}".format(thres))
        logger.info("minDCF (p_target:{} c_miss:{} c_fa:{}) = {:.4f}".format(
            args.p_target, args.c_miss, args.c_fa, min_dcf))


if __name__ == "__main__":
    main()
