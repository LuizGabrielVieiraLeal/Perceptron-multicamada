#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 12:59:20 2018

@author: Gabriel Vieira
"""

import numpy as np


#Operador XOR
data = np.array([
            [0, 0], 
            [0, 1], 
            [1, 0], 
            [1, 1]
        ]) #Entradas

outputs = np.array([[0], [1], [1], [0]]) #Saídas

synapsesToHiddenLayer = 2 * np.random.random((2, 3)) - 1 #Pesos da entrada para a camada oculta

synapsesToOutput = 2 * np.random.random((3, 1)) - 1 #Pesos da camada oculta para a camada de saída

trainingTime = 100000 #Épocas (rodadas de atualização dos pesos)

learning_rate = 0.3 #Taxa de aprendizagem

time = 1 #Momento

def sigmoid_function(sumResult):
    """
    Função de ativação - retorna valores entre 0 e 1
    """
    return 1 / (1 + np.exp(- sumResult))

def sigmoid_derivative(sigmoidResult):
    """
    Cálculo da derivada da função sigmoid - direciona o lado da curvatura do gradiente para atualizar os pesos
    """
    return sigmoidResult * (1 - sigmoidResult)

for j in range(trainingTime):
    """
    Loop de treinamento da rede
    """
    dataLayer = data #Variável auxiliar para não manipular o dados diretamente
    
    #Alimentando camada oculta
    sumToHiddenLayer = np.dot(dataLayer, synapsesToHiddenLayer)
    hiddenLayerResults = sigmoid_function(sumToHiddenLayer)
    
    #Alimentando camada de saída
    sumToOutput = np.dot(hiddenLayerResults, synapsesToOutput)
    outputLayerResults = sigmoid_function(sumToOutput)
    
    outputLayerErrors = outputs - outputLayerResults #Calculando erros
    absoluteAverage = np.mean(np.abs(outputLayerErrors)) #Calculando média absoluta dos erros
    
    #Percentual de acerto
    if j == trainingTime -1:
        print('Percentual de acerto atual da rede neural {0:.2f}%'.format((1 - absoluteAverage) * 100))
    
    #Cálculo da descida de gradiente
    
    #Obtendo a variação entre os erros e as derivações
    outputLayerDelta = outputLayerErrors * sigmoid_derivative(outputLayerResults)
    
    #Transpondo a matriz de synapses para realizar o produto escalar das matrizes
    synapseToOutputTransposed = synapsesToOutput.T
    #Obtendo a variação entre os erros e as derivações
    hiddenLayerDelta = (outputLayerDelta.dot(synapseToOutputTransposed)) * sigmoid_derivative(hiddenLayerResults)
    
    #Ajuste das sinapses (backpropagation)
    
    #Ajuste das sinapses da camada oculta para a camada de saída
    hiddenLayerResultsTransposed = hiddenLayerResults.T
    
    newSynapsesToOutput = hiddenLayerResultsTransposed.dot(outputLayerDelta)
    synapsesToOutput = synapsesToOutput * time + (newSynapsesToOutput * learning_rate)
    
    #Ajuste das sinapses da camada de entrada para a camada oculta
    dataLayerTransposed = dataLayer.T
    
    newSynapsesToHiddenLayer = dataLayerTransposed.dot(hiddenLayerDelta)
    synapsesToHiddenLayer = synapsesToHiddenLayer * time + (newSynapsesToHiddenLayer * learning_rate)
    
    
    
    
    