#!\usr\bin\env python
import pathlib
import numpy as np

inputpath = pathlib.Path('v0')
cmpenv = np.load(inputpath / 'cmpenv.npy', allow_pickle=True)
print(type(cmpenv))
print(cmpenv)
