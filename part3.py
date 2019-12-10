import os
import sys
import numpy as np
from part2 import preprocess, get_count, do_smoothing, get_emission_param, evalResult

def get_transition_param(data, k):
  """
  find q(v|u) = count(u -> v) / count(u)
  """

  # uses part 2's functions (hence the imports)
  data = preprocess(data)
  count_x, count_y = get_count(data)
  count_x = do_smoothing(count_x, k)

  eol = count_y['<eol>']
  del count_y['<eol>']
  count_y['<eol>'] = eol

  # how many unique x and y are there
  # use it to initialize np.zeros
  total_x = len(count_x.keys())
  total_y = len(count_y.keys())
  count_u_v = np.zeros((total_y, total_y), dtype="float") # use only y because u and v are states

  # conversion between i => x or y and vice versa x or y => i
  i2x = list(count_x)
  i2y = list(count_y)
  x2i = {x: i for i, x in enumerate(i2x)}
  y2i = {y: i for i, y in enumerate(i2y)}

  # fill in transition params
  # add one here, because the iteration will go up to len - 1 and last one is <eol>
  count_u_v[y2i["<eol>"], y2i[data[0][1]]] += 1
  for i in range(len(data)-1):
      count_u_v[y2i[data[i][1]], y2i[data[i+1][1]]] += 1

  # do q(v|u) = count(u -> v) / count(u) for each u which is y
  q = count_u_v
  for i in range(total_y):
    q[i] = count_u_v[i] / count_y[i2y[i]]

  return q, i2x, i2y, x2i, y2i

def viterbi(sentence, e, q, y2i, x2i, i2y, i2x):

  n_tag = len(i2y)-1
  pi = np.zeros((e.shape[0], len(sentence)))              # table with maximum value of each node
  parents = np.full((e.shape[0], len(sentence)-1 ), -1)   # the parent of the maximum value

  # the first node
  pi[:n_tag,0] = q[y2i["<eol>"], :n_tag] + e[:n_tag, x2i.get(sentence[0], x2i['#UNK#'])]

  # dynamically compute highest probability
  for i in range(1, len(sentence)):
    t_cost = pi[:, i-1].reshape(-1,1) + q
    parents[:n_tag, i-1] = np.argmax(t_cost[:n_tag,:n_tag], axis=0)

    pi[:n_tag, i] = np.max(t_cost[:n_tag,:n_tag], axis=0) + e[:n_tag, x2i.get(sentence[i], x2i['#UNK#'])]

  # the last node
  t_cost = pi[:,len(sentence)-1].reshape(-1,1) + q
  parent = np.argmax(t_cost[:n_tag,n_tag], axis=0)
  tag_seq = [i2y[parent]]

  for i in range(len(sentence)-1,0,-1):
    parent = parents[parent,i-1]
    tag_seq.append(i2y[parent])

  return tag_seq[::-1]

def predict_all_y(params, e, in_path, out_path):
  q = params[0]
  i2x = params[1]
  i2y = params[2]
  x2i = params[3]
  y2i = params[4]

  # fix underflow
  e = np.log(e+0.000001)
  q = np.log(q+0.000001)

  dev_in = open(in_path, "r", encoding="utf-8").read().splitlines()
  with open(out_path,'w', encoding="utf-8") as f_result:
      sentence = []
      for x in dev_in:
          if x == '':
              all_y = viterbi(sentence, e, q, y2i, x2i, i2y, i2x)
              f_result.write('\n'.join(['{} {}'.format(w,t) for w,t in zip(sentence,all_y)]))
              f_result.write('\n\n')
              sentence = []
              # break
          else:
              sentence.append(x)

  return out_path

def run(smoothing_k):
  k = smoothing_k

  AL = os.path.join("AL", "train")
  e = get_emission_param(AL, k)[0]
  params = get_transition_param(AL, k)
  AL_in = os.path.join("AL", "dev.in")
  AL_out = os.path.join("AL", "dev.p3.out")
  AL_out = predict_all_y(params, e, AL_in, AL_out)
  print(f"Finished writing {AL_out}\n")

  CN = os.path.join("CN", "train")
  e = get_emission_param(CN, k)[0]
  params = get_transition_param(CN, k)
  CN_in = os.path.join("CN", "dev.in")
  CN_out = os.path.join("CN", "dev.p3.out")
  CN_out = predict_all_y(params, e, CN_in, CN_out)
  print(f"Finished writing {CN_out}\n")

  EN = os.path.join("EN", "train")
  e = get_emission_param(EN, k)[0]
  params = get_transition_param(EN, k)
  EN_in = os.path.join("EN", "dev.in")
  EN_out = os.path.join("EN", "dev.p3.out")
  EN_out = predict_all_y(params, e, EN_in, EN_out)
  print(f"Finished writing {EN_out}\n")

  SG = os.path.join("SG", "train")
  e = get_emission_param(SG, k)[0]
  params = get_transition_param(SG, k)
  SG_in = os.path.join("SG", "dev.in")
  SG_out = os.path.join("SG", "dev.p3.out")
  SG_out = predict_all_y(params, e, SG_in, SG_out)
  print(f"Finished writing {SG_out}\n")

if __name__ == "__main__":
  smoothing_k = 3

  gold_path = [ os.path.join("AL", "dev.out"),
                os.path.join("CN", "dev.out"),
                os.path.join("EN", "dev.out"),
                os.path.join("SG", "dev.out") ]

  prediction_path = [ os.path.join("AL", "dev.p3.out"),
                      os.path.join("CN", "dev.p3.out"),
                      os.path.join("EN", "dev.p3.out"),
                      os.path.join("SG", "dev.p3.out") ]

  run(smoothing_k)
  evalResult(gold_path, prediction_path)
  print(f"\nThis is default run with smoothing_k = {smoothing_k}")
  sys.exit()


