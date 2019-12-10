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


  n_tag = len(i2y)-1

  # now holds k values from 1st to kth top
  pi = np.full((k, e.shape[0], len(sentence)),np.NINF)
  parents = np.full((k, e.shape[0], len(sentence)-1), -1)
  pi[0, :n_tag, 0] = q[y2i['<eol>'], :n_tag] + e[:n_tag, x2i.get(sentence[0], x2i['#UNK#'])]

  for i in range(1,len(sentence)):
    t_cost = pi[:,:,i-1:i]+np.tile(q,(k,1,1))
    t_cost = np.concatenate(t_cost,axis=0)

    idx = np.argsort(t_cost,axis=0)[::-1]
    parents[:,:n_tag,i-1] = idx[:k,:n_tag]%(n_tag+1)

    for j in range(n_tag):
        pi[:,j,i] = t_cost[:,j][idx[:k,j]]+e[j,x2i.get(sentence[i],x2i['#UNK#'])]

  t_cost = pi[:,:,len(sentence)-1:len(sentence)]+np.tile(q,(k,1,1))
  t_cost = np.concatenate(t_cost,axis=0)[:,n_tag]
  idx = np.argsort(t_cost,axis=0)[::-1][:k]
  idx = idx%(n_tag+1)
  tag_seq = [i2y[idx[-1]]]
  parent = idx[-1]
  rank = np.sum(idx==idx[-1])

  for i in range(len(sentence)-1,0,-1):
    new_parent = parents[rank-1,parent,i-1]
    rank = np.sum(parents[:,parent,i-1][:rank]==new_parent)
    parent = new_parent
    tag_seq.append(i2y[parent])

  return tag_seq[::-1]

def predict_all_y(k, params, e, in_path, out_path):
  q = params[0]
  # fix underflow
  e = np.log(e+0.000001)
  q = np.log(q+0.000001)

  dev_in = open(in_path, "r", encoding="utf-8").read().splitlines()
  with open(out_path,'w', encoding="utf-8") as f_result:
      sentence = []
      for x in dev_in:
          if x == '':
              all_y = k_viterbi(k, sentence, e, q, params)
              f_result.write('\n'.join(['{} {}'.format(w,t) for w,t in zip(sentence,all_y)]))
              f_result.write('\n\n')
              sentence = []
              # break
          else:
              sentence.append(x)

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
