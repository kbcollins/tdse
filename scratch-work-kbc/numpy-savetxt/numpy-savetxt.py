# import pathlib
import numpy as np

filehandle = open('./test.txt', mode='a')

# the open().write() method only takes one argument
# and it must be  a string. You have to put in your own
# line breaks with '\n' if you want them
delim=', '
filehandle.write('0.1' + delim)
filehandle.write('-2' + delim)
# last element in row, give new line and not delimiter
filehandle.write('1.7e-4' + '\n')

# what happens when we try to save a python list?
# - it saves in the exact way that you would expect
#   the list to be output with a print() statement
testlist = [1, 2, 3.0]
filehandle.write(str(testlist) + '\n')

# numpy.savetxt() is good for saving numpy objects
# but open().write() gives more flexibility
# You can save in the same file with both
# open().write() and numpy.savetxt(), you can
# even save on the same line with both statements
test = [0.1, -2, 1.7e-4]
testarry = np.array([test, test, test])
testarry[1] = testarry[1] *2
print(testarry)
np.savetxt(filehandle, testarry, delimiter='\t')


# close file before ending program
filehandle.close()