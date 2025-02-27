import torch
import random
import numpy as np
import torch.utils.data
import logging
from collections import defaultdict

from colorama import Fore
from torchvision import datasets, transforms
from tqdm import tqdm

logger = logging.getLogger("logger")


class MSPDataloader():

    def __init__(self, params):
        self.params = params
        if self.params['load_data_from_pkl'] == True:
            pre_cached_data = torch.load(self.params['pre_cache_data_path'])
            self.train_dataloader = pre_cached_data['train_dataset']
            self.test_dataloader = pre_cached_data['test_dataset']

        else:
            self.load_dataset()

    def load_dataset(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_emnist = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])
        transform_grstb = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transform_grstb_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if self.params["dataset"].upper() == "CIFAR10":
            self.train_dataset = datasets.CIFAR10(f"{self.params['data_dir']}", train=True, download=True,
                                                  transform=transform_train)
            self.test_dataset = datasets.CIFAR10(f"{self.params['data_dir']}", train=False, download=True,
                                                 transform=transform_test)

        elif self.params["dataset"].upper() == "CIFAR100":
            self.train_dataset = datasets.CIFAR100(f"{self.params['data_dir']}", train=True, download=True,
                                                   transform=transform_train)
            self.test_dataset = datasets.CIFAR100(f"{self.params['data_dir']}", train=False, download=True,
                                                  transform=transform_test)
        elif self.params["dataset"].upper() == "GTSRB":
            # self.train_dataset = datasets.EMNIST(f"{self.params['data_dir']}", train=True, split="mnist", download=True,
            #                                      transform=transform_emnist)
            # self.test_dataset = datasets.EMNIST(f"{self.params['data_dir']}", train=False, split="mnist", transform=transform_emnist)
            self.train_dataset = datasets.GTSRB(f"{self.params['data_dir']}", split="train", download=True,
                                                transform=transform_grstb)
            self.train_dataset = [sample for sample in self.train_dataset] * 3
            self.test_dataset = datasets.GTSRB(f"{self.params['data_dir']}", split="test", download=True,
                                               transform=transform_grstb_test)


        elif self.params["dataset"].upper() == "EMNIST":
            # self.train_dataset = datasets.EMNIST(f"{self.params['data_dir']}", train=True, split="mnist", download=True,
            #                                      transform=transform_emnist)
            # self.test_dataset = datasets.EMNIST(f"{self.params['data_dir']}", train=False, split="mnist", transform=transform_emnist)
            self.train_dataset = datasets.MNIST(f"{self.params['data_dir']}", train=True, download=True,
                                                transform=transform_emnist)
            self.test_dataset = datasets.MNIST(f"{self.params['data_dir']}", train=False, transform=transform_emnist)

        indices_per_participant = self.sample_dirichlet_train_data(
            self.params['no_of_total_participants'],
            alpha=self.params['dirichlet_alpha'])
        from torch.utils.data import Subset
        train_loaders = []

        for pos, indices in tqdm(indices_per_participant.items()):
            tmp_subset = Subset(self.train_dataset, indices)
            train_loader = torch.utils.data.DataLoader(
                tmp_subset,
                batch_size=self.params["train_batch_size"],
                shuffle=True,
                drop_last=True)
            train_loaders.append(train_loader)

        self.train_dataloader = train_loaders

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params["test_batch_size"],
            shuffle=False, drop_last=True)

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
