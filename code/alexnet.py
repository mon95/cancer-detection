from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
import json
import csv
from pathlib import Path

from optimizer import Optimizer
from classifier import Classifier
from dataset import DataSet

'''
File: alexnet.py
Project: cs6476-computervision-project
File Created: October 2019
Author: Shalini Chaudhuri (you@you.you)
'''

if __name__ == '__main__':

    data_dir = 'data'
    model_name = 'alexnet'
    output_classes = 2
    feature_extract = True

    paramsFile = open('./code/paramsAlexnet.json')
    paramsMetricFilePath = Path('./modelEvaluationMetrics/alexnet/alexnetMetrics.csv')
    hyperparamsArray = json.load(paramsFile)
    fields = ['learningRate', 'momentum', 'epochs', 'batchSize', 'valAccuracy']
    metricsPath = './modelEvaluationMetrics/alexnet/'

    best_accuracy_until_now = 0

    # if not paramsMetricFilePath.exists():
    #     with open(paramsMetricFilePath, 'a+') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(fields) 

    # for ctr, hyperparams in enumerate(hyperparamsArray):
    #     learningRate = hyperparams['learningRate']
    #     epochs = hyperparams['epochs']
    #     momentum = hyperparams['momentum']
    #     batchSize = hyperparams['batchSize']

    #     # read learning rate, momentum, batchSize and numEpochs from predefined csv
    #     # store each separately
    #     # batch_size = 8
    #     # num_epochs = 15
    #     # learningRate = 0.001
    #     # momentum = 0.9
            
    #     alexnetClassifier = Classifier(model_name, output_classes)
    #     model = alexnetClassifier.initPretrainedModel(224)

    # #     print(model)

    #     dataloaders_dict, dataset_sizes, class_names = DataSet.initDataLoaders(data_dir, batchSize)
    #     data_transforms = DataSet.setUpDataLoaderTransformers()
        

    #     image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}

    #     # Detect if we have a GPU available
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    #     sgdOptimizer = Optimizer(device)
    #     optimizer_ft = sgdOptimizer.optimize(model, feature_extract, learningRate, momentum)

    #     criterion = nn.CrossEntropyLoss()

    #     model, train_acc_history, val_acc_history, best_accuracy = alexnetClassifier.train_model(model, 
    #             criterion, 
    #             optimizer_ft, 
    #             dataloaders_dict, 
    #             dataset_sizes)



    #     with open(paramsMetricFilePath, 'a+') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([learningRate, epochs, momentum, batchSize, best_accuracy]) 

    #     plt.title("Validation Accuracy vs. Number of Epochs")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Validation Accuracy")
    #     plt.plot(range(1,epochs+1), val_acc_history)
    #     plt.clf()
    #     plt.cla()

    #     plt.savefig(metricsPath+'val_epoch_'+str(ctr)+'.png')
    #     np.save(metricsPath+'val_epoch_'+str(ctr)+'.npy', val_acc_history)


    #     plt.title("Training Accuracy vs. Number of Epochs")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Training Accuracy")
    #     plt.plot(range(1,epochs+1), train_acc_history)
    #     plt.clf()
    #     plt.cla()

    #     plt.savefig(metricsPath+'train_epoch_'+str(ctr)+'.png')
    #     np.save(metricsPath+'train_epoch_'+str(ctr)+'.npy', train_acc_history)
        
    #     if best_accuracy > best_accuracy_until_now:
    #         print(best_accuracy)
    #         best_accuracy_until_now = best_accuracy

    #         # save only the best model until now
    #         torch.save({
    #                 'name': 'alexnetFeatureExtraction',
    #                 'epoch': epochs,
    #                 'learningRate': learningRate,
    #                 'momentum': momentum,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer_ft.state_dict(),
    #                 }, './trainedModels/alexnetFeatureExtraction.pt')



#     # testing
    alexnetClassifier = Classifier(model_name, output_classes)
    model = alexnetClassifier.initPretrainedModel(224)
    batchSize = 8
    dataloaders_dict, dataset_sizes, class_names = DataSet.initDataLoaders(data_dir, batchSize)
    data_transforms = DataSet.setUpDataLoaderTransformers()
    state = torch.load('./trainedModels/alexnetFeatureExtraction.pt')
    model.load_state_dict(state['model_state_dict'])
    predictions = alexnetClassifier.testModel(dataloaders_dict, model, class_names, dataset_sizes, batchSize)

    # save predicted values
    np.savetxt('./trainedModels/predictedLabelsAlexNet.csv', predictions, fmt='%s')


        
