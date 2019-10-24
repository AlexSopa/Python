import numpy as np
a = np.arange(15).reshape(3, 5)
#What does this look like?
print(a)
#How many axis does this have?
print('There is %d axis' % a.ndim)
print(('The shape of a is {}').format(a.shape))
#Creating a zero array
zeros = np.zeros((10,10), dtype=np.int16) #dtype decides what kind of number it is(int16 is an int, the default is float I believe)
print(zeros)
#What is the shape of this one?
print(zeros.shape)
#If you want an array instead of a list do this stuff
arraysRbetter = np.array([10,5,2,3,4,2,])
print(('Arrays are just this much better >> {}').format(arraysRbetter))
#Asking for numbers for an array
#five_num = input("Give me 5 numbers seperated by commans ex : 1,2,3,4,5")
#This sends the message and waits for input
#type(five_num)
#five_num should now be = to the 5 numbers they gave or whatever else they typed in
#More types of arrays you can make
a2 = np.ones( (2,3,4), dtype=np.int16)
print(a2)
