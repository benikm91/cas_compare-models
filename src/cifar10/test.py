import pickle
from pathlib import Path

with open('result.pickle', 'rb') as f:
    result = pickle.load(f)

with open('cifar10_20220130-175841.pickle', 'rb') as f:
    cifar10_20220130 = pickle.load(f)

result['CNN'] = cifar10_20220130['CNN']

print(list(result.keys()))

with open(f"result2.pickle", 'wb') as file:
    pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
