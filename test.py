import numpy
import sys
import timeit
print( 100.4 // 2)
a = (20, 1, 5, 5)
b = (2, 2)
print(a[1:])
print(numpy.prod(a[1:]))
print(a[0] * numpy.prod(a[2:]))
fan_out = (a[0] * numpy.prod(a[2:]) // numpy.prod(b))
print(fan_out)
print(numpy.random.RandomState(23455).rand(3))
print(numpy.random.RandomState(1233).rand(3))
print(numpy.random.RandomState(23455).rand(3))
print(sys.getdefaultencoding())