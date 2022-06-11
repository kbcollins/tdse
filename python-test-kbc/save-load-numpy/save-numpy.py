#! /usr/bin/env python
import sys
from pathlib import Path
import numpy as np

# currentdir = Path(sys.argv[1]).parent
# modulename = Path(sys.argv[1]).parts[-1]

currentdir = Path()

print(currentdir)
print(currentdir.resolve())

from function import fn

saveme = fn(7.65)
print(saveme)

np.save(currentdir/'saved-data/saved_np_array', saveme)