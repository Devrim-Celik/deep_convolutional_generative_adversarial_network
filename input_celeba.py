import pickle
import numpy as np

def load_celeba(filename = "data/pickle/celeba_pickle.dat"):
  with open(filename,"rb") as f:

    data = pickle.load(f)
    print('T2')
    print(data[0].shape)
    for i in range(len(data)):
      data[i] = (data[i]/255.0 - 0.5) * 2.0
      data[i] = np.expand_dims(data[i],-1)
    return data