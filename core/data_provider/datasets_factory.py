from torchvision import transforms
from torch.utils.data import DataLoader

def data_provider(dataset, configs, path, batch_size, mode, is_shuffle=True):
    if mode == 'train':
        num_workers = configs.num_workers
    else:
        num_workers = 0

    if dataset == 'mnist':
        from core.data_provider.mnist import mnist as data_set
        from core.data_provider.mnist import ToTensor, Norm
    elif dataset == 'town':
        from core.data_provider.towncentre import towncentre as data_set
        from core.data_provider.towncentre import ToTensor, Norm
    elif dataset == 'sun':
        from core.data_provider.sun import sun as data_set
        from core.data_provider.sun import ToTensor

    dataset = data_set(
        configs=configs,
        path=path,
        mode=mode,
        transform=transforms.Compose([ToTensor()]))

    return DataLoader(dataset, pin_memory=True, drop_last=True, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers)
