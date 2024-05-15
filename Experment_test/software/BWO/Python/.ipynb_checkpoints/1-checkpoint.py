
import numpy
import random
from numpy.random import gamma
import math

ub=[2 ,2]
lb = [1,1]
lb = numpy.asarray(lb)
ub = numpy.asarray(ub)
SearchAgents_no=2
dim=2
aee = numpy.asarray([pos * (ub - lb) + lb for pos in numpy.random.uniform(0, 1, (SearchAgents_no, dim))])
print(aee[0,:])