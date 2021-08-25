"""
HACKy implementation
"""

import argparse
import copy
import os
import random
import time

import numpy as np
from sklearn.metrics import recall_score, precision_score

import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from distil.active_learning_strategies import GLISTER
from distil.utils.data_handler import DataHandler_ChestXRayImage
from dataloader import ChestXRayImageData, ChestXRayImageDataView
from arch import XRayNet


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Pneumonia classifier for ChestXRay Images")
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('save_model_path', type=str)
    parser.add_argument('--budget', type=float)
    parser.add_argument('--no_training', action="store_true")
    parser.add_argument('--distil', action="store_true")
    parser.add_argument('--strategy', type=str)
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    if args.verbosity:
        print(f"Args: {args}")
    return args


def get_acc(model, dloader):

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        ypred = []
        ytruth = []
        for data in dloader:
            images, labels = data['image'], data['label']
            outputs = model(images.to("cuda:0")).cpu()
            _, predicted = torch.max(torch.nn.functional.log_softmax(outputs.data, dim=1), 1)
            ytruth.extend(labels.tolist())
            ypred.extend(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Recall: {recall_score(ytruth, ypred)}")
    print(f"Precision: {precision_score(ytruth, ypred)}")
    return 100 * correct / total


def train(model, optimizer, objective, dloader, epochs, writer):
    # HACK
    def weight_reset(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    model.apply(weight_reset)
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")
        model.train()
        etrain(model, optimizer, objective, dloader, epoch, writer)
        lrScheduler.step()


def etrain(model, optimizer, objective, dloader, epoch, writer):
    running_loss = 0.0
    for step, data in enumerate(dloader):
        imgs, labels = data['image'], data['label']

        optimizer.zero_grad()

        pred = model(imgs.to("cuda:0"))
        loss = objective(pred, labels.to("cuda:0"))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 10 == 9:
            writer.add_scalar("training loss", running_loss / 10, epoch * len(dloader) + step)
            running_loss = 0.0


if __name__ == "__main__":
    torch.cuda.set_device(0)
    args = parse_args()
    writer = SummaryWriter(args.save_model_path)

    trainDataset = ChestXRayImageData(f"{args.dataset_path}/train/")
    testDataset = ChestXRayImageData(f"{args.dataset_path}/test/")

    # HACK
    budget = 500
    start_with = 500
    num_rounds = 10
    num_classes = 2

    model = XRayNet(3, num_classes)

    model = model.to("cuda:0")

    if args.distil:

        loader = torch.utils.data.DataLoader(trainDataset, batch_size=len(trainDataset))
        iterator = iter(loader)
        fullTrainData = next(iterator)
        del iterator, loader

        idx = np.random.choice(len(trainDataset), size=start_with, replace=False)
        #currentSubset = ChestXRayImageDataView(trainDataset)
        #currentSubset.set_view(idx)
        Xtr = fullTrainData['image'][idx]
        ytr = fullTrainData['label'][idx]
        Xunlabeled = np.delete(fullTrainData['image'], idx, axis=0)
        yunlabeled = np.delete(fullTrainData['label'], idx, axis=0)

        strategy = GLISTER(
            Xtr,
            ytr,
            Xunlabeled,
            model,
            DataHandler_ChestXRayImage,
            num_classes,
            {"batch_size": 20, "lr": 1e-2},
            valid=False,
            typeOf="Diversity",
            lam=10
        )
    else:
        num_rounds = 1

    print(f"Model summary: {model}")
    if not args.no_training:
        print(f"Training for {num_rounds} rounds with {args.epochs} epochs ...")

        #reduction = "none" if args.distil else "mean"
        reduction = "mean"

        objective = torch.nn.CrossEntropyLoss(reduction=reduction, weight=torch.cuda.FloatTensor([1.95, 0.67]))

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
        lrScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_loss = float('inf')

        if args.distil:
            for i in range(num_rounds):
                print(f"Training round: {i}")
                if i != 0:
                    idx = strategy.select(budget)
                    #previousSubset = currentSubset
                    #currentSubset = ChestXRayImageDataView(trainDataset)
                    #idx = previousSubset.view + idx
                    #currentSubset.set_view(idx)
                    Xtr = np.concatenate((Xtr, Xunlabeled[idx]), axis=0)
                    ytr = np.concatenate((ytr, yunlabeled[idx]), axis=0)
                    Xunlabeled = np.delete(Xunlabeled, idx, axis=0)
                    yunlabeled = np.delete(yunlabeled, idx, axis=0)

                strategy.update_data(Xtr, ytr, Xunlabeled)
                currentSubset = DataHandler_ChestXRayImage(Xtr, ytr, select=False, return_index=False, return_dict=True)
                print(f"Size of current labeled (approx) training set: {len(currentSubset)}")
                currentSubsetLoader = torch.utils.data.DataLoader(
                    currentSubset,
                    batch_size=args.batch_size,
                    shuffle=True
                )
                train(model, optimizer, objective, currentSubsetLoader, args.epochs, writer)
                shadowModel = copy.deepcopy(model)
                strategy.update_model(shadowModel)
                testDataloader = torch.utils.data.DataLoader(
                    testDataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=3
                )
                print(f"Accuracy: {get_acc(model, testDataloader)}")
        else:
            gammas = None
            trainDataLoader = torch.utils.data.DataLoader(
                trainDataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=3
            )
            train(model, optimizer, objective, trainDataLoader, args.epochs, writer)
            testDataloader = torch.utils.data.DataLoader(
                testDataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=3
            )
            print(f"Accuracy: {get_acc(model, testDataloader)}")


        print('Finished Training')
    else:
        print(f"No training ...")
