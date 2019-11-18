import os
import numpy as np


def load_data(fpath):
  cwd = os.getcwd()
  fpath = os.path.join(cwd, fpath)
  file = open(fpath, "r")
  return file.readlines()

class HiddenMarkovModel():
  def __init__(self, data):
    '''
    builds 2 dictionary yx and y
    yx :
        key is "<token> <tag>"  : value is count of this combination
        Market I-NP             : 20

    y :
        key is "<tag>"  : value is total count of tag
        I-NP            : 200

    x :
        key is "<token>"  : value is total count of token
        Market            : 20
    '''
    print("\nInitializing Hidden Markov Model...")
    self.yx = {}
    self.y = {}
    self.x = {}

    for chunks in data:
      self.yx[chunks.strip()] = self.yx.get(chunks.strip(), 0) + 1
      if len(chunks.split()) == 2:
        self.y[chunks.split()[1]] = self.y.get(chunks.split()[1], 0) + 1
        self.x[chunks.split()[0]] = self.x.get(chunks.split()[0], 0) + 1

    self.count_yx = sum(self.yx.values())
    self.count_y = sum(self.y.values())
    self.count_x = sum(self.x.values())

    print(f"Populated total of {self.count_yx} y -> x")
    print(f"Populated total of {self.count_x} x")
    print(f"Populated total of {self.count_y} y: {self.y}")


  def get_emission_parameter(self):
    '''
    e(x|y) = count(y -> x) / count(y)
    saved e(x|y) as
    '''
    self.e = {}
    for chunks in self.yx:
      count_yx = self.yx[chunks]
      # for some reason there is this key-value: ":7663"
      if len(chunks.split()) == 2:
        count_y = self.y[chunks.split()[1]]

      self.e[chunks] = count_yx / count_y

    return self.e

  def do_smoothing(self, k=3):
    '''
    replaces (y -> x) that has count(y -> x) < k
    with #UNK# += count(y -> x), preserves total count(y -> x)
    '''
    to_be_deleted = []
    count_UNK = 0

    for chunks in self.yx:
      if self.yx[chunks] < k:
        to_be_deleted.append(chunks)
        count_yx = self.yx[chunks]
        count_UNK += count_yx

    self.yx["#UNK#"] = count_UNK

    for chunks in to_be_deleted:
      del self.yx[chunks]

    self.count_yx = sum(self.yx.values())
    print(f"Finished smoothing with k={k}. Count(y -> x) preserved at {self.count_yx}")
    return self.yx

  def predict_y(self, test_data):
    ans = []
    for token in test_data:
      print(token.strip())


if __name__ == "__main__":
  data = load_data(os.path.join("EN", "train"))
  HMM_EN = HiddenMarkovModel(data)

  k = 3
  HMM_EN_yx = HMM_EN.do_smoothing(k)

  HMM_EN_e = HMM_EN.get_emission_parameter()
  # print(HMM_EN_e)

  # do prediction
  test_data = load_data(os.path.join("EN", "dev.in"))
  HMM_EN.predict_y(test_data)

