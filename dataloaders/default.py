import torch
import torchvision
from torchvision import transforms
from .sampler import RandSubClassSampler

import numpy as np


def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignment_index(assign_0, assign_1, col):
    new_index = assign_0 * col + assign_1
    return new_index


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data

def IndianPines(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1


    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return train_iter, test_iter






def MNIST(batch_sz, num_workers=2):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

    train_dataset = torchvision.datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 10

    eval_dataset = torchvision.datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 10

    return train_loader, eval_loader

def CIFAR10(batch_sz, num_workers=2):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 10

    test_dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 10

    return train_loader, eval_loader


def CIFAR100(batch_sz, num_workers=2):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_dataset = torchvision.datasets.CIFAR100(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 100

    test_dataset = torchvision.datasets.CIFAR100(
        root='data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 100

    return train_loader, eval_loader

def Omniglot(batch_sz, num_workers=2):
    # This dataset is only for training the Similarity Prediction Network on Omniglot background set
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    train_dataset = torchvision.datasets.Omniglot(
        root='data', download=True, background=True,
        transform=transforms.Compose(
           [transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize]
        ))
    train_length = len(train_dataset)
    train_imgid2cid = [train_dataset[i][1] for i in range(train_length)]  # train_dataset[i] returns (img, cid)
    # Randomly select 20 characters from 964. By default setting (batch_sz=100), each character has 5 images in a mini-batch.
    train_sampler = RandSubClassSampler(
        inds=range(train_length),
        labels=train_imgid2cid,
        cls_per_batch=20,
        batch_size=batch_sz,
        num_batch=train_length//batch_sz)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=False,
                                               num_workers=num_workers, sampler=train_sampler)
    train_loader.num_classes = 964

    test_dataset = torchvision.datasets.Omniglot(
        root='data', download=True, background=False,
        transform=transforms.Compose(
          [transforms.Resize(32),
           transforms.ToTensor(),
           binary_flip,
           normalize]
        ))
    eval_length = len(test_dataset)
    eval_imgid2cid = [test_dataset[i][1] for i in range(eval_length)]
    eval_sampler = RandSubClassSampler(
        inds=range(eval_length),
        labels=eval_imgid2cid,
        cls_per_batch=20,
        batch_size=batch_sz,
        num_batch=eval_length // batch_sz)
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False,
                                              num_workers=num_workers, sampler=eval_sampler)
    eval_loader.num_classes = 659

    return train_loader, eval_loader


def omniglot_alphabet_func(alphabet, background):
    def create_alphabet_dataset(batch_sz, num_workers=2):
        # This dataset is only for unsupervised clustering
        # train_dataset (with data augmentation) is used during the optimization of clustering criteria
        # test_dataset (without data augmentation) is used after the clustering is converged

        binary_flip = transforms.Lambda(lambda x: 1 - x)
        normalize = transforms.Normalize((0.086,), (0.235,))

        train_dataset = torchvision.datasets.Omniglot(
            root='data', download=True, background=background,
            transform=transforms.Compose(
               [transforms.RandomResizedCrop(32, (0.85, 1.)),
                transforms.ToTensor(),
                binary_flip,
                normalize]
            ))

        # Following part dependents on the internal implementation of official Omniglot dataset loader
        # Only use the images which has alphabet-name in their path name (_characters[cid])
        valid_flat_character_images = [(imgname,cid) for imgname,cid in train_dataset._flat_character_images if alphabet in train_dataset._characters[cid]]
        ndata = len(valid_flat_character_images)  # The number of data after filtering
        train_imgid2cid = [valid_flat_character_images[i][1] for i in range(ndata)]  # The tuple (valid_flat_character_images[i]) are (img, cid)
        cid_set = set(train_imgid2cid)  # The labels are not 0..c-1 here.
        cid2ncid = {cid:ncid for ncid,cid in enumerate(cid_set)}  # Create the mapping table for New cid (ncid)
        valid_characters = {cid2ncid[cid]:train_dataset._characters[cid] for cid in cid_set}
        for i in range(ndata):  # Convert the labels to make sure it has the value {0..c-1}
            valid_flat_character_images[i] = (valid_flat_character_images[i][0],cid2ncid[valid_flat_character_images[i][1]])

        # Apply surgery to the dataset
        train_dataset._flat_character_images = valid_flat_character_images
        train_dataset._characters = valid_characters

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True,
                                                   num_workers=num_workers)
        train_loader.num_classes = len(cid_set)

        test_dataset = torchvision.datasets.Omniglot(
            root='data', download=True, background=background,
            transform=transforms.Compose(
              [transforms.Resize(32),
               transforms.ToTensor(),
               binary_flip,
               normalize]
            ))

        # Apply surgery to the dataset
        test_dataset._flat_character_images = valid_flat_character_images  # Set the new list to the dataset
        test_dataset._characters = valid_characters

        eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False,
                                                  num_workers=num_workers)
        eval_loader.num_classes = train_loader.num_classes

        print('=> Alphabet %s has %d characters and %d images.'%(alphabet, train_loader.num_classes, len(train_dataset)))
        return train_loader, eval_loader
    return create_alphabet_dataset

omniglot_evaluation_alphabets_mapping = {
    'Malayalam':'Malayalam',
     'Kannada':'Kannada',
     'Syriac':'Syriac_(Serto)',
     'Atemayar_Qelisayer':'Atemayar_Qelisayer',
     'Gurmukhi':'Gurmukhi',
     'Old_Church_Slavonic':'Old_Church_Slavonic_(Cyrillic)',
     'Manipuri':'Manipuri',
     'Atlantean':'Atlantean',
     'Sylheti':'Sylheti',
     'Mongolian':'Mongolian',
     'Aurek':'Aurek-Besh',
     'Angelic':'Angelic',
     'ULOG':'ULOG',
     'Oriya':'Oriya',
     'Avesta':'Avesta',
     'Tibetan':'Tibetan',
     'Tengwar':'Tengwar',
     'Keble':'Keble',
     'Ge_ez':'Ge_ez',
     'Glagolitic':'Glagolitic'
}

# Create the functions to access the individual alphabet dataset in Omniglot
for funcName, alphabetStr in omniglot_evaluation_alphabets_mapping.items():
    locals()['Omniglot_eval_' + funcName] = omniglot_alphabet_func(alphabet=alphabetStr, background=False)
