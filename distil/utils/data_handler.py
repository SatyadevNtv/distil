from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import torch 

class DataHandler_Points(Dataset):
    """
    Data Handler to load data points.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """
    def __init__(self, X, Y=None, select=True, use_test_transform=False,return_index=True):
        """
        Constructor
        """
        
        self.select = select
        if not self.select:
        	self.X = X.astype(np.float32)
        	self.Y = Y
        else:
        	self.X = X.astype(np.float32)  #For unlabeled Data

        self.return_index = return_index

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            
            if self.return_index:
                return x, y, index
            else:
                return x, y
        else:
            x = self.X[index]              #For unlabeled Data
            if self.return_index:
                return x, index
            else:
                return x

    def __len__(self):
        return len(self.X)

class DataHandler_SVHN(Dataset):
    """
    Data Handler to load SVHN dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    use_test_transform: bool, optional
        Use test transform without augmentations like crop, flip, etc. (default: False)
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False,return_index=True):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.test_gen_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
        if not self.select:
            self.X = X
            self.Y = Y
        else:
            self.X = X

        self.return_index = return_index

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, y, index
            else:
                return x, y

        else:
            x = self.X[index]
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, index
            else:
                return x

    def __len__(self):
        return len(self.X)

class DataHandler_MNIST(Dataset):
    """
    Data Handler to load MNIST dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    image_dim: int, optional
        dimension of the input image (32 for LeNet, 28 for MNISTNet)
    select: bool
        True if loading data without labels, False otherwise
    use_test_transform: bool, optional
        Use test transform without augmentations like crop, flip, etc. (default: False)
    duplicateChannels: bool, optional
        Duplicate channels for black and white images
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False, image_dim=28, duplicateChannels=False):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.duplicateChannels = duplicateChannels
        self.training_gen_transform = transforms.Compose([transforms.RandomCrop(image_dim, padding=4), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.test_gen_transform = transforms.Compose([transforms.Resize((image_dim, image_dim)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if not self.select:
            self.X = X
            self.Y = Y
        else:
            self.X = X

        self.return_index = return_index

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)

            if(x.shape[0]==1 and self.duplicateChannels): x = torch.repeat_interleave(x, 3, 0)
            return x, y, index
        else:
            x = self.X[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
                
            if(x.shape[0]==1 and self.duplicateChannels): x = torch.repeat_interleave(x, 3, 0)
            return x, index


    def __len__(self):
        return len(self.X)
    
class DataHandler_KMNIST(Dataset):
    """
    Data Handler to load KMNIST dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    use_test_transform: bool, optional
        Use test transform without augmentations like crop, flip, etc. (default: False)

    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False,return_index=True):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.test_gen_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Use mean/std of MNIST
        if not self.select:
            self.X = X
            self.Y = Y
        else:
            self.X = X

        self.return_index = return_index

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, y, index
            else:
                return x, y

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)

            if self.return_index:
                return x, index
            else:
                return x

    def __len__(self):
        return len(self.X)

class DataHandler_FASHION_MNIST(Dataset):
    """
    Data Handler to load FASHION_MNIST dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    use_test_transform: bool, optional
        Use test transform without augmentations like crop, flip, etc. (default: False)
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False,return_index=True):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.test_gen_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # Use mean/std of MNIST
        if not self.select:
            self.X = X
            self.Y = Y
        else:
            self.X = X

        self.return_index = return_index

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, y, index
            else:
                return x, y

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, index
            else:
                return x

    def __len__(self):
        return len(self.X)

class DataHandler_CIFAR10(Dataset):
    """
    Data Handler to load CIFAR10 dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    use_test_transform: bool, optional
        Use test transform without augmentations like crop, flip, etc. (default: False)
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False,return_index=True):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.test_gen_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if not self.select:
            self.X = X
            self.Y = Y
        else:
            self.X = X

        self.return_index = return_index

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, y, index
            else:
                return x, y

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, index
            else:
                return x

    def __len__(self):
        return len(self.X)
    
class DataHandler_CIFAR100(Dataset):
    """
    Data Handler to load CIFAR100 dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    use_test_transform: bool, optional
        Use test transform without augmentations like crop, flip, etc. (default: False)
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False,return_index=True):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.test_gen_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        if not self.select:
            self.X = X
            self.Y = Y
        else:
            self.X = X

        self.return_index = return_index

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, y, index
            else:
                return x, y

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, index
            else:
                return x

    def __len__(self):
        return len(self.X)
    
class DataHandler_STL10(Dataset):
    """
    Data Handler to load STL10 dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    use_test_transform: bool, optional
        Use test transform without augmentations like crop, flip, etc. (default: False)
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False,return_index=True):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.RandomCrop(96, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.test_gen_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
        if not self.select:
            self.X = X
            self.Y = Y
        else:
            self.X = X

        self.return_index = return_index

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, y, index
            else:
                return x, y

        else:
            x = self.X[index]
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            
            if self.return_index:
                return x, index
            else:
                return x

    def __len__(self):
        return len(self.X)

class DataHandler_ChestXRayImage(Dataset):

    def __init__(self, X, y=None, select=True, return_index=True, return_dict=False):
        self.select = select
        if not self.select:
            self.X = X
            self.Y = y
        else:
            self.X = X
        self.return_index = return_index
        self.return_dict = return_dict


    def __getitem__(self, idx):
        vals = []
        vals.append(self.X[idx])
        if not self.select:
            vals.append(self.Y[idx])
        if self.return_index:
            vals.append(idx)

        if self.return_dict:
            return {"image": vals[0], "label": vals[1]}

        return tuple(vals)

    def __len__(self):
        return len(self.X)
