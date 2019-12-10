import os
import sys
import numpy as np
from part2 import preprocess, get_count, do_smoothing, get_emission_param, evalResult
from part3 import get_transition_param

def k_viterbi(k, sentence, e, q, params):
  i2x = params[1]
  i2y = params[2]
  x2i = params[3]
  y2i = params[4]

  n = len(i2y)-1

  # now holds k values from 1st to kth top
  pi = np.full((k, e.shape[0], len(sentence)),np.NINF)
  previous = np.full((k, e.shape[0], len(sentence)-1), -1)
  pi[0, :n, 0] = q[y2i['<eol>'], :n] + e[:n, x2i.get(sentence[0], x2i['#UNK#'])]

  for i in range(1,len(sentence)):
    alpha = pi[:,:,i-1:i]+np.tile(q,(k,1,1))
    alpha = np.concatenate(alpha,axis=0)

    idx = np.argsort(alpha,axis=0)[::-1]
    previous[:,:n,i-1] = idx[:k,:n]%(n+1)

    for j in range(n):
        pi[:,j,i] = alpha[:,j][idx[:k,j]]+e[j,x2i.get(sentence[i],x2i['#UNK#'])]

  alpha = pi[:,:,len(sentence)-1:len(sentence)]+np.tile(q,(k,1,1))
  alpha = np.concatenate(alpha,axis=0)[:,n]
  idx = np.argsort(alpha,axis=0)[::-1][:k]
  idx = idx%(n+1)
  all_y = [i2y[idx[-1]]]
  prev = idx[-1]
  rank = np.sum(idx==idx[-1])

  for i in range(len(sentence)-1,0,-1):
    new_prev = previous[rank-1,prev,i-1]
    rank = np.sum(previous[:,prev,i-1][:rank]==new_prev)
    prev = new_prev
    all_y.append(i2y[prev])

  return all_y[::-1]

def predict_all_y(k, params, e, in_path, out_path):
  q = params[0]
  e = np.log(e + 0.000001)
  q = np.log(q + 0.000001)

  dev_in = open(in_path, "r", encoding="utf-8").read().splitlines()
  dev_out = open(out_path,'w', encoding="utf-8")

  sentence = []
  for x in dev_in:
    if x == '':
      all_y = k_viterbi(k, sentence, e, q, params)
      dev_out.write('\n'.join([f'{x} {y}' for x, y in zip(sentence,all_y)]))
      dev_out.write('\n\n')
      sentence = []
    else: sentence.append(x)
  return out_path

def run(smooth_k, viterbi_k):
  smoothing_k = smooth_k
  k = viterbi_k

  EN = os.path.join("EN", "train")
  e = get_emission_param(EN, smoothing_k)[0]
  params = get_transition_param(EN, smoothing_k)
  EN_in = os.path.join("EN", "dev.in")
  EN_out = os.path.join("EN", "dev.p4.out")
  EN_out = predict_all_y(k, params, e, EN_in, EN_out)
  print(f"Finished writing {EN_out}\n")

  AL = os.path.join("AL", "train")
  e = get_emission_param(AL, smoothing_k)[0]
  params = get_transition_param(AL, smoothing_k)
  AL_in = os.path.join("AL", "dev.in")
  AL_out = os.path.join("AL", "dev.p4.out")
  AL_out = predict_all_y(k, params, e, AL_in, AL_out)
  print(f"Finished writing {AL_out}\n")

  CN = os.path.join("CN", "train")
  e = get_emission_param(CN, smoothing_k)[0]
  params = get_transition_param(CN, smoothing_k)
  CN_in = os.path.join("CN", "dev.in")
  CN_out = os.path.join("CN", "dev.p4.out")
  CN_out = predict_all_y(k, params, e, CN_in, CN_out)
  print(f"Finished writing {CN_out}\n")

  SG = os.path.join("SG", "train")
  e = get_emission_param(SG, smoothing_k)[0]
  params = get_transition_param(SG, smoothing_k)
  SG_in = os.path.join("SG", "dev.in")
  SG_out = os.path.join("SG", "dev.p4.out")
  SG_out = predict_all_y(k, params, e, SG_in, SG_out)
  print(f"Finished writing {SG_out}\n")


if __name__ == "__main__":
  smoothing_k = 3
  viterbi_k = 7

  gold_path = [ os.path.join("AL", "dev.out"),
                os.path.join("CN", "dev.out"),
                os.path.join("EN", "dev.out"),
                os.path.join("SG", "dev.out") ]

  prediction_path = [ os.path.join("AL", "dev.p4.out"),
                      os.path.join("CN", "dev.p4.out"),
                      os.path.join("EN", "dev.p4.out"),
                      os.path.join("SG", "dev.p4.out") ]

  run(smoothing_k, viterbi_k)
  evalResult(gold_path, prediction_path)
  print(f"\nThis is default run with smoothing_k = {smoothing_k} and viterbi_k = {viterbi_k}")
  sys.exit()
