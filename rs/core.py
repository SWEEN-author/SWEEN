'''
References:
[1] J. Cohen, E. Rosenfeld and Z. Kolter. 
Certified Adversarial Robustness via Randomized Smoothing. In ICML, 2019.
[2] Zhai, Runtian, et al. 
MACER: Attack-Free and Scalable Robust Training via Maximizing Certified Radius. In ICLR, 2020.

Acknowledgements:
[1] https://github.com/locuslab/smoothing/blob/master/code/core.py
[2] https://github.com/RuntianZ/macer/blob/master/rs/core.py
'''

from math import ceil

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import torch


class Smooth(object):
  '''
  Smoothed classifier
  beta is the inverse of softmax temperature
  '''

  # to abstain, Smooth returns this int
  ABSTAIN = -1

  def __init__(self, base_classifier: torch.nn.Module, num_classes: int,
               sigma: float, device, beta=1.0, adp=False):
      self.base_classifier = base_classifier
      self.num_classes = num_classes
      self.sigma = sigma
      self.device = device
      self.square_sum = None
      self.beta = beta
      self.adp = adp

  def certify(self, x: torch.tensor, n0: int, n: int,
              alpha: float, batch_size: int):
      if self.adp:
          total_eval_num = 0
          total_input_num = 0
          # make an initial prediction of the label
          cAHat, e_n, i_n = self.predict(x, n0, batch_size)
          total_eval_num += e_n
          total_input_num += i_n
          # draw more samples of f(x + epsilon)
          observation, e_n, i_n = self._sample_noise(x, n, batch_size)
          total_eval_num += e_n
          total_input_num += i_n
      else:
          # make an initial prediction of the label
          cAHat = self.predict(x, n0, batch_size)
          # draw more samples of f(x + epsilon)
          observation = self._sample_noise(x, n, batch_size)
          
      # use these samples to estimate a lower bound on pA
      nA = observation[cAHat].item()
      pABar = self._lower_confidence_bound(nA, n, alpha)
      if pABar < 0.5:
          if self.adp:
              return Smooth.ABSTAIN, 0.0, total_eval_num, total_input_num
          return Smooth.ABSTAIN, 0.0
      else:
          radius = self.sigma * norm.ppf(pABar)
          if self.adp:
              return cAHat, radius, total_eval_num, total_input_num
          return cAHat, radius

  def predict(self, x: torch.tensor, n: int, batch_size: int):
      self.base_classifier.eval()
      if self.adp:
          result, e_n, i_n = self._sample_noise(x, n, batch_size)
          return result.argsort()[::-1][0], e_n, i_n
      else:
          result = self._sample_noise(x, n, batch_size)
          return result.argsort()[::-1][0]

  def _sample_noise(self, x: torch.tensor, num: int, batch_size):
      with torch.no_grad():
          result_hard = np.zeros(self.num_classes, dtype=int)
          self.square_sum = np.zeros(self.num_classes, dtype=float)
          if self.adp:
              total_eval_num = 0
              total_input_num = 0
          for _ in range(ceil(num / batch_size)):
              this_batch_size = min(batch_size, num)
              num -= this_batch_size
              
              batch = x.repeat((this_batch_size, 1, 1, 1))
              noise = torch.randn_like(batch, device=self.device) * self.sigma
              if self.adp:
                  predictions, eval_num = self.base_classifier(batch + noise)
                  total_eval_num += eval_num
                  total_input_num += this_batch_size
              else:
                  predictions = self.base_classifier(batch + noise)
              
              predictions *= self.beta
              
              p_hard = predictions.argmax(1)
              result_hard += self._count_arr(p_hard.cpu().numpy(),self.num_classes)
          if self.adp:
              return result_hard, total_eval_num, total_input_num
          return result_hard


  def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
      counts = np.zeros(length, dtype=int)
      for idx in arr:
          counts[idx] += 1
      return counts

  def _lower_confidence_bound(self, NA, N, alpha: float) -> float:
      return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
