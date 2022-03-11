import pybrain3
import pickle
import csv
import matplotlib.pylab as plt
from numpy import ravel
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.xml.networkreader import NetworkReader
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.structure import SoftmaxLayer
from pybrain3.structure import TanhLayer


fileObject = open('Mature dump H1RDBD.txt', 'rb')
net2 = pickle.load(fileObject)
NetworkWriter.writeToFile(net2, 'H1RDCD_Network.xml')
fileObject.close()
# # Conditions
# y = net2.activate([1, 1, 1])
# print('All resources, All Devops, Max backlog changes', y)
#
# y = net2.activate([1, 1, 0])
# print('All resources, All Devops, No backlog changes', y)
#
# y = net2.activate([0, 0, 1])
# print('No resources, No Devops, Max backlog changes', y)
#
# y = net2.activate([0, 1, 1])
# print('No resources, All Devops, Max backlog changes', y)
#
# y = net2.activate([0, 0, 0])
# print('No resources, No Devops, No backlog changes', y)
#
# y = net2.activate([1, 0, 1])
# print('All resources, No Devops, Max backlog changes', y)
#
# y = net2.activate([1, 0, 0])
# print('All resources, No Devops, No backlog changes', y)



