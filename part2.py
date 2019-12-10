import os
import sys
import numpy as np
from evalResultMod import get_predicted, get_observed, printResult, compare_observed_to_predicted, separator, outputColumnIndex

def preprocess(path):
  """
  takes in raw data convert into:
  [['Municipal', 'B-NP'], ['bonds', 'I-NP'],..., ["", "<eol>"], ... ]
  """
  raw = open(path, "r", encoding="utf-8").read().splitlines()

  data = []
  for chunk in raw:
    if chunk != "": data.append(chunk.split())
    else: data.append(["", "<eol>"])
    # <eol> marks end of line

  return data

def get_count(data):
  """
  get:
  count(x) => words/observations
  count(y) => states
  """

  count_x = {}
  count_y = {}

  # [[x, y], ...<eol>, ...] is [["Municipal", "B-VP"], ..., ["", "<eol>"], ...]
  for chunk in data:
    x = chunk[0]
    y = chunk[1]
    count_x[x] = count_x.get(x, 0) + 1
    count_y[y] = count_y.get(y, 0) + 1

  return count_x, count_y

def do_smoothing(count_x, k):
  """
  if count_x[x] < k:
  delete it and replace with #UNK#
  """

  to_be_deleted = []
  count_UNK = 0
  for x, count in count_x.items():
    if count < k:
      count_UNK += count
      to_be_deleted.append(x)
    else: continue

  # add count_UNK and
  # delete those entries less than k
  count_x["#UNK#"] = count_UNK
  for x in to_be_deleted: del count_x[x]

  return count_x

def get_emission_param(data, k):
  """
  find e(x|y) = count(y -> x) / count(y)
  """

  # calls the previous functions to parse raw and get_count
  data = preprocess(data)
  count_x, count_y = get_count(data)
  count_x = do_smoothing(count_x, k)

  # how many unique x and y are there
  # use it to initialize np.zeros
  total_x = len(count_x.keys())
  total_y = len(count_y.keys())
  count_y_x = np.zeros((total_y, total_x), dtype="float")

  # conversion between i => x or y and vice versa x or y => i
  i2x = list(count_x)
  i2y = list(count_y)
  x2i = {x: i for i, x in enumerate(i2x)}
  y2i = {y: i for i, y in enumerate(i2y)}

  # fill in emission params
  for i in range(len(data)):
    if data[i][0] == "<eol>": continue
    else:
      count_y_x[y2i[data[i][1]], x2i.get(data[i][0], x2i['#UNK#'])] += 1

  # do e(x|y) = count(y -> x) / count(y) for each y
  e = count_y_x
  for i in range(total_y):
    e[i] = count_y_x[i] / count_y[i2y[i]]

  return e, i2x, i2y, x2i, y2i

def get_e_argmax(e):
  best_y = np.argmax(e, axis=0)
  return best_y

def predict_y(params, in_path, out_path):
  """
  given a test set, label all x observations with a tag y
  if x cannot be found in the dictionary, then replace with #UNK#
  return the path of the file
  """

  # unpack params
  e = params[0]
  i2x = params[1]
  i2y = params[2]
  x2i = params[3]
  y2i = params[4]

  dev_in = open(in_path, "r", encoding="utf-8").read().splitlines()
  dev_out = open(out_path, "w", encoding="utf-8")

  best_y = get_e_argmax(e)
  # go line by line of test dev_in and label and write to dev_out
  for x in dev_in:
    if x != "":
      if x2i.get(x, -1) == -1: # could not find x, word = #UNK#
        x = "#UNK#"
      y = i2y[best_y[x2i[x]]]
      dev_out.write(f"{x} {y}\n")

    else: dev_out.write(f"{x}\n")
    # if empty line, just copy empty line cause it's the end of sentence

  dev_out.close()
  return out_path

def run(k):
  smoothing_k = k

  # AL
  AL = os.path.join("AL", "train")
  AL_params = get_emission_param(AL, smoothing_k)
  print(f"AL emission:\n{AL_params[0]}")
  AL_in = os.path.join("AL", "dev.in")
  AL_out = os.path.join("AL", "dev.p2.out")
  AL_out = predict_y(AL_params, AL_in, AL_out)
  print(f"Finished writing {AL_out}\n")

  # CN
  CN = os.path.join("CN", "train")
  CN_params = get_emission_param(CN, smoothing_k)
  print(f"CN emission:\n{CN_params[0]}")
  CN_in = os.path.join("CN", "dev.in")
  CN_out = os.path.join("CN", "dev.p2.out")
  CN_out = predict_y(CN_params, CN_in, CN_out)
  print(f"Finished writing {CN_out}\n")

  # EN
  EN = os.path.join("EN", "train")
  EN_params = get_emission_param(EN, smoothing_k)
  print(f"EN emission:\n{EN_params[0]}")
  EN_in = os.path.join("EN", "dev.in")
  EN_out = os.path.join("EN", "dev.p2.out")
  EN_out = predict_y(EN_params, EN_in, EN_out)
  print(f"Finished writing {EN_out}\n")

  # SG
  SG = os.path.join("SG", "train")
  SG_params = get_emission_param(SG, smoothing_k)
  print(f"SG emission:\n{SG_params[0]}")
  SG_in = os.path.join("SG", "dev.in")
  SG_out = os.path.join("SG", "dev.p2.out")
  SG_out = predict_y(SG_params, SG_in, SG_out)
  print(f"Finished writing {SG_out}\n")

def evalResult(gold_path, prediction_path):
  print("\n==========================\n")
  for i in range(len(gold_path)):
    print(f"\nChecking {prediction_path[i]}...")
    gold = open(gold_path[i], "r", encoding='UTF-8')
    prediction = open(prediction_path[i], "r", encoding='UTF-8')

    #column separator
    separator = ' '

    #the column index for tags
    outputColumnIndex = 1
    #Read Gold data
    observed = get_observed(gold)

    #Read Predction data
    predicted = get_predicted(prediction)

    #Compare
    compare_observed_to_predicted(observed, predicted)

# for unit testing
if __name__ == "__main__":
  smoothing_k = 3

  gold_path = [ os.path.join("AL", "dev.out"),
                os.path.join("CN", "dev.out"),
                os.path.join("EN", "dev.out"),
                os.path.join("SG", "dev.out") ]

  prediction_path = [ os.path.join("AL", "dev.p2.out"),
                      os.path.join("CN", "dev.p2.out"),
                      os.path.join("EN", "dev.p2.out"),
                      os.path.join("SG", "dev.p2.out") ]

  run(smoothing_k)
  evalResult(gold_path, prediction_path)
  print(f"\nThis is default run with smoothing_k = {smoothing_k}")
  sys.exit()
