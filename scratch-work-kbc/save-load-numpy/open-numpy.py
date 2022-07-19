#! /usr/bin/env python
from pathlib import Path
import numpy as np

currentdir = Path()

validationarray = np.linspace(0, 7.65)

loadedarray = np.load(currentdir/'saved-data/saved_np_array.npy')

print(np.linalg.norm(validationarray - loadedarray))