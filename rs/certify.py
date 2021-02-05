'''
References:
[1] J. Cohen, E. Rosenfeld and Z. Kolter. 
Certified Adversarial Robustness via Randomized Smoothing. In ICML, 2019.
[2] Zhai, Runtian, et al. 
MACER: Attack-Free and Scalable Robust Training via Maximizing Certified Radius. In ICLR, 2020.

Acknowledgements:
[1] https://github.com/locuslab/smoothing/blob/master/code/certify.py
[2] https://github.com/RuntianZ/macer/blob/master/rs/certify.py
'''

import numpy as np
from time import time
import datetime

from rs.core import Smooth


def certify(model, device, dataset, transform, num_classes, outdir,
            start_img=0, num_img=500, skip=1, sigma=0.25, N0=100, N=100000,
            alpha=0.001, batch=1000, 
            grid=(0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25), beta=1.0, adp=False):
    print('===certify(N={}, sigma={})==='.format(N, sigma))
    model.eval()
    # create the smooothed classifier g
    smoothed_classifier = Smooth(model, num_classes, sigma, device, beta=beta,adp=adp)
    radius_hard = np.zeros((num_img,), dtype=np.float)
    num_grid = len(grid)
    cnt_grid = np.zeros((num_grid + 1,), dtype=np.int)
    s = 0.0
    if adp:
        total_eval_num = 0
        total_input_num = 0
    # prepare output file
    f = open(outdir, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    
    # iterate through the dataset
    for i in range(num_img):
        img, target = dataset[start_img + i * skip]
        img = img.to(device)    
        before_time = time()
        # certify the prediction of g around x
        if adp:
            prediction, radius, e_n, i_n = smoothed_classifier.certify(img, N0, N, alpha, batch)
            total_eval_num += e_n
            total_input_num += i_n
        else:
            prediction, radius = smoothed_classifier.certify(img, N0, N, alpha, batch)
        after_time = time()
        correct = int(prediction == target)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            start_img + i * skip, target, prediction, radius, correct, time_elapsed), file=f, flush=True)
        
        radius_hard[i] = radius if correct == 1 else -1
        if correct == 1:
            cnt_grid[0] += 1
            s += radius
            for j in range(num_grid):
                if radius >= grid[j]:
                    cnt_grid[j + 1] += 1

        
    f.close()
    print('===Certify Summary===')
    print('Total Image Number: {}'.format(num_img))
    print('Radius: 0.0  Number: {}  Acc: {}'.format(cnt_grid[0], cnt_grid[0] / num_img * 100))
    for j in range(num_grid):
        print('Radius: {}  Number: {}  Acc: {}'.format(
                grid[j], cnt_grid[j + 1], cnt_grid[j + 1] / num_img * 100))
    print('ACR: {}'.format(s / num_img))
    if adp:
        print('Total Eval Number:{}, Total Input Number:{}'.format(total_eval_num, total_input_num))
        print('Average Eval Number:{}'.format(total_eval_num/total_input_num))