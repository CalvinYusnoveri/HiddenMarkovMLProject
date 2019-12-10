import os
import sys
import numpy as np
from part2 import preprocess, get_count, do_smoothing, evalResult, get_emission_param
from part3 import get_transition_param

def get_forward_backward_emission(data, k):
  """
  find forward emission and backward emission
  """
  # calls the previous functions to parse raw and get_count
  data = preprocess(data)
  count_x, count_y = get_count(data)
  count_x = do_smoothing(count_x, k)

  # how many unique x and y are there
  # use it to initialize np.zeros
  total_x = len(count_x.keys())
  total_y = len(count_y.keys())
  count_y_x_forward = np.zeros((total_y, total_x), dtype="float")
  count_y_x_back = np.zeros((total_y, total_x), dtype="float")

  eol = count_y['<eol>']
  del count_y['<eol>']
  count_y['<eol>'] = eol

  # conversion between i => x or y and vice versa x or y => i
  i2x = list(count_x)
  i2y = list(count_y)
  x2i = {x: i for i, x in enumerate(i2x)}
  y2i = {y: i for i, y in enumerate(i2y)}

  # except for beginning and last node, count ef and eb
  for i in range(1, len(data)-1):
    count_y_x_forward[y2i[data[i-1][1]], x2i.get(data[i][0], x2i['#UNK#'])] += 1
    count_y_x_back[y2i[data[i][1]], x2i.get(data[i-1][0], x2i['#UNK#'])] += 1

  ef = count_y_x_forward
  eb = count_y_x_back
  for i in range(total_y):
    ef[i] = count_y_x_forward[i] / count_y[i2y[i]]
    eb[i] = count_y_x_back[i] / count_y[i2y[i]]

  return eb, ef

def modified_viterbi(sentence, e, ef, eb, q, y2i, x2i, i2y, i2x):

  e = np.log(e + 0.00001)
  eb = np.log(eb + 0.00001)
  ef = np.log(ef + 0.00001)
  q = np.log(q + 0.00001)

  # initialize node values and pointer
  pi = np.zeros((e.shape[0], len(sentence)))
  previous = np.full((e.shape[0], len(sentence)-1 ), -1)

  # number of tags (y)
  n = len(i2y) - 1

  # the first node
  pi[:n, 0] = q[y2i["<eol>"], :n] + e[:n, x2i.get(sentence[0], x2i['#UNK#'])]

  for i in range(1,len(sentence)):
    alpha = pi[:,i-1].reshape(-1,1) + q
    # add in eb
    alpha += eb[:, x2i.get(sentence[i-1], x2i['#UNK#'])]
    # add in ef except for last one
    if i != (len(sentence)-1):
        alpha += ef[:, x2i.get(sentence[i+1], x2i['#UNK#'])]

    previous[:n,i-1] = np.argmax(alpha[:n,:n],axis=0)
    pi[:n, i] = np.max(alpha[:n,:n], axis=0) + e[:n, x2i.get(sentence[i],x2i['#UNK#'])]

  # the last node
  alpha = pi[:, len(sentence)-1].reshape(-1,1) + q
  prev = np.argmax(alpha[:n,n], axis=0)
  all_y = [i2y[prev]]

  for i in range(len(sentence)-1, 0, -1):
    prev = previous[prev, i-1]
    all_y.append(i2y[prev])

  return all_y[::-1]

def predict_all_y(params, e, ef, eb, in_path, out_path):

  q = params[0]
  i2x = params[1]
  i2y = params[2]
  x2i = params[3]
  y2i = params[4]

  dev_in = open(in_path, "r", encoding="utf-8").read().splitlines()
  dev_out = open(out_path,'w', encoding="utf-8")

  sentence = []
  for x in dev_in:
    if x == '':
      all_y = modified_viterbi(sentence, e, ef, eb, q, y2i, x2i, i2y, i2x)
      dev_out.write('\n'.join([f'{x} {y}' for x, y in zip(sentence,all_y)]))
      dev_out.write('\n\n')
      sentence = []
    else: sentence.append(x)
  return out_path

def run(smoothing_k):
  k = smoothing_k

  AL = os.path.join("AL", "train")
  eb, ef = get_forward_backward_emission(AL, k)
  e = get_emission_param(AL, k)[0]
  params = get_transition_param(AL, k)
  AL_in = os.path.join("AL", "dev.in")
  AL_out = os.path.join("AL", "dev.p5.out")
  AL_out = predict_all_y(params, e, ef, eb, AL_in, AL_out)
  print(f"Finished writing {AL_out}\n")

  test1_in = os.path.join("Test", "AL", "test.in")
  test1_out = os.path.join("Test", "AL", "test.p5.out")
  test1_out = predict_all_y(params, e, ef, eb, test1_in, test1_out)
  print(f"Finished writing {test1_out}\n")

  CN = os.path.join("CN", "train")
  eb, ef = get_forward_backward_emission(CN, k)
  e = get_emission_param(CN, k)[0]
  params = get_transition_param(CN, k)
  CN_in = os.path.join("CN", "dev.in")
  CN_out = os.path.join("CN", "dev.p5.out")
  CN_out = predict_all_y(params, e, ef, eb, CN_in, CN_out)
  print(f"Finished writing {CN_out}\n")

  EN = os.path.join("EN", "train")
  eb, ef = get_forward_backward_emission(EN, k)
  e = get_emission_param(EN, k)[0]
  params = get_transition_param(EN, k)
  EN_in = os.path.join("EN", "dev.in")
  EN_out = os.path.join("EN", "dev.p5.out")
  EN_out = predict_all_y(params, e, ef, eb, EN_in, EN_out)
  print(f"Finished writing {EN_out}\n")

  test2_in = os.path.join("Test", "EN", "test.in")
  test2_out = os.path.join("Test", "EN", "test.p5.out")
  test2_out = predict_all_y(params, e, ef, eb, test2_in, test2_out)
  print(f"Finished writing {test2_out}\n")

  SG = os.path.join("SG", "train")
  eb, ef = get_forward_backward_emission(SG, k)
  e = get_emission_param(SG, k)[0]
  params = get_transition_param(SG, k)
  SG_in = os.path.join("SG", "dev.in")
  SG_out = os.path.join("SG", "dev.p5.out")
  SG_out = predict_all_y(params, e, ef, eb, SG_in, SG_out)
  print(f"Finished writing {SG_out}\n")


if __name__ == "__main__":
  smoothing_k = 3

  gold_path = [ os.path.join("AL", "dev.out"),
                os.path.join("CN", "dev.out"),
                os.path.join("EN", "dev.out"),
                os.path.join("SG", "dev.out") ]

  prediction_path = [ os.path.join("AL", "dev.p5.out"),
                      os.path.join("CN", "dev.p5.out"),
                      os.path.join("EN", "dev.p5.out"),
                      os.path.join("SG", "dev.p5.out") ]

  run(smoothing_k)
  evalResult(gold_path, prediction_path)
  print(f"\nThis is default run with smoothing_k = {smoothing_k}")
  sys.exit()
