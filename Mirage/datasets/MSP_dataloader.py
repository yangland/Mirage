import torch
import random
import numpy as np
import torch.utils.data
import logging
from collections import defaultdict
import os
from colorama import Fore
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
logger = logging.getLogger("logger")


# class MSPDataloader():

#     def __init__(self, params):
#         self.params = params
#         if self.params['load_data_from_pkl'] == True:
#             pre_cached_data = torch.load(self.params['pre_cache_data_path'])
#             self.train_dataloader = pre_cached_data['train_dataset']
#             self.test_dataloader = pre_cached_data['test_dataset']

#         else:
#             self.load_dataset()

#     def load_dataset(self):
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])


#         transform_test = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])

#         transform_emnist = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.ToTensor()
#         ])
#         transform_grstb = transforms.Compose([
#             transforms.Resize((32, 32)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(10),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
#         transform_grstb_test = transforms.Compose([
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])

#         if self.params["dataset"].upper() == "CIFAR10":
#             self.train_dataset = datasets.CIFAR10(f"{self.params['data_dir']}", train=True, download=True,
#                                                   transform=transform_train)
#             self.test_dataset = datasets.CIFAR10(f"{self.params['data_dir']}", train=False, download=True,
#                                                  transform=transform_test)

#         elif self.params["dataset"].upper() == "CIFAR100":
#             self.train_dataset = datasets.CIFAR100(f"{self.params['data_dir']}", train=True, download=True,
#                                                    transform=transform_train)
#             self.test_dataset = datasets.CIFAR100(f"{self.params['data_dir']}", train=False, download=True,
#                                                   transform=transform_test)
#         elif self.params["dataset"].upper() == "GTSRB":
#             self.train_dataset = datasets.GTSRB(f"{self.params['data_dir']}", split="train", download=True,
#                                                 transform=transform_grstb)
#             self.train_dataset = [sample for sample in self.train_dataset] * 3
#             self.test_dataset = datasets.GTSRB(f"{self.params['data_dir']}", split="test", download=True,
#                                                transform=transform_grstb_test)


#         elif self.params["dataset"].upper() == "EMNIST":
#             self.train_dataset = datasets.MNIST(f"{self.params['data_dir']}", train=True, download=True,
#                                                 transform=transform_emnist)
#             self.test_dataset = datasets.MNIST(f"{self.params['data_dir']}", train=False, transform=transform_emnist)

#         indices_per_participant = self.sample_dirichlet_train_data(
#             self.params['no_of_total_participants'],
#             alpha=self.params['dirichlet_alpha'])
#         
#         train_loaders = []

#         for pos, indices in tqdm(indices_per_participant.items()):
#             tmp_subset = Subset(self.train_dataset, indices)
#             train_loader = torch.utils.data.DataLoader(
#                 tmp_subset,
#                 batch_size=self.params["train_batch_size"],
#                 shuffle=True,
#                 drop_last=True)
#             train_loaders.append(train_loader)

#         self.train_dataloader = train_loaders

#         self.test_dataloader = torch.utils.data.DataLoader(
#             self.test_dataset,
#             batch_size=self.params["test_batch_size"],
#             shuffle=False, drop_last=True)



class MSPDataloader():
    def __init__(self, params):
        self.params = params
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = self._resolve_data_dir()
        
        if self.params.get('load_data_from_pkl', False):
            self._load_precached_data()
        else:
            self.load_dataset()

    def _resolve_data_dir(self):
        """Ensure data directory exists and return absolute path"""
        data_dir = os.path.join(self.project_root, self.params.get('data_dir', 'data'))
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def _load_precached_data(self):
        """Load pre-cached data from pickle file"""
        try:
            pre_cached_data = torch.load(self.params['pre_cache_data_path'])
            self.train_dataloader = pre_cached_data['train_dataset']
            self.test_dataloader = pre_cached_data['test_dataset']
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-cached data: {str(e)}")

    def _get_transforms(self):
        """Define all dataset transforms"""
        return {
            'cifar': {
                'train': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            },
            'emnist': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor()
            ]),
            'gtsrb': {
                'train': transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]),
                'test': transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            }
        }

    def load_dataset(self):
        """Load the specified dataset"""
        transforms = self._get_transforms()
        dataset_name = self.params["dataset"].upper()

        try:
            if dataset_name == "CIFAR10":
                self._load_cifar10(transforms['cifar'])
            elif dataset_name == "CIFAR100":
                self._load_cifar100(transforms['cifar'])
            elif dataset_name == "GTSRB":
                self._load_gtsrb(transforms['gtsrb'])
            elif dataset_name == "EMNIST":
                self._load_emnist(transforms['emnist'])
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            self._create_data_loaders()
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

    def _load_cifar10(self, transforms):
        self.train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms['train']
        )
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transforms['test']
        )

    def _load_cifar100(self, transforms):
        self.train_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms['train']
        )
        self.test_dataset = datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transforms['test']
        )

    def _load_gtsrb(self, transforms):
        self.train_dataset = datasets.GTSRB(
            root=self.data_dir,
            split="train",
            download=True,
            transform=transforms['train']
        )
        self.train_dataset = [sample for sample in self.train_dataset] * 3
        self.test_dataset = datasets.GTSRB(
            root=self.data_dir,
            split="test",
            download=True,
            transform=transforms['test']
        )

    def _load_emnist(self, transform):
        self.train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transform
        )

    def _create_data_loaders(self):
        """Create data loaders for federated learning"""
        indices_per_participant = self.sample_dirichlet_train_data(
            self.params['no_of_total_participants'],
            alpha=self.params['dirichlet_alpha']
        )

        train_loaders = []
        for pos, indices in tqdm(indices_per_participant.items(), desc="Creating participant loaders"):
            train_loaders.append(
                DataLoader(
                    Subset(self.train_dataset, indices),
                    batch_size=self.params["train_batch_size"],
                    shuffle=True,
                    drop_last=True
                )
            )

        self.train_dataloader = train_loaders
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.params["test_batch_size"],
            shuffle=False,
            drop_last=True
        )


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
