import numpy as np
import cv2 as cv
import os
import copy
import pathlib
import tensorflow as tf
import keras
import torch
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.layers import Input, Dense
import matplotlib.pyplot as plt
from utils import *

device = tf.device('cpu')

train = LoadData("train")
test = LoadData("test")

class ConvNet(tf.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(4, (1, 1), activation='relu', input_shape=(1,1,50,100)))
        model.add(layers.MaxPooling2D((1, 1)))



    def forward(self,x):
        x = model.layers.linear(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


num_epochs = 10
bigTest = []
bigTrain = []

def trainModel():
    model = ConvNet().to(device)
    criterion = tf.keras.losses.MSE()
    optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam', **kwargs
)

    bestModel = model
    bestScore = 10000
    testscores = []
    trainscores = []

    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        np.random.shuffle(train)

        for i,(im, label) in enumerate(train):


            output = model(im)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                # testSc = evaluateModel(model,test,sidelen=900)
                testSc = Evaluate(model,test)
                # trainSc = evaluateModel(model,trainingSet,sidelen=900)
                trainSc = Evaluate(model,train)
                if testSc < bestScore:
                    bestModel = copy.deepcopy(model)
                    bestScore = testSc
                testscores.append(testSc)
                trainscores.append(trainSc)

                print(trainSc)
                print(testSc)
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, len(train), loss.item()))

    bigTest.append(testscores)
    bigTrain.append(trainscores)

    finalScore = Evaluate(bestModel,test)
    print(finalScore)

    if finalScore < 150:
        torch.save(bestModel.state_dict(), "yModels/" + str(int(finalScore))+".plt")
for i in range(6):
    trainModel()