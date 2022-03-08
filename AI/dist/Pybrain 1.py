import pybrain3
from pybrain3.tools.shortcuts import buildNetwork
net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])
print('Y=', y)
print(net)
a = net['bias']
print(a)