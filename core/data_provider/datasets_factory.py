from torchvision import transforms
from torch.utils.data import DataLoader


def data_provider(dataset, configs, data_train_path, data_test_path, batch_size,
                  is_training=True,
                  is_shuffle=True):
    if dataset == 'mnist':
        from core.data_provider.mnist import mnist as data_set
        from core.data_provider.mnist import ToTensor, Norm
    elif dataset == 'kitti':
        from core.data_provider.KITTI import KITTI as data_set
        from core.data_provider.KITTI import ToTensor, Norm
    elif dataset == 'town':
        from core.data_provider.towncentre import towncentre as data_set
        from core.data_provider.towncentre import ToTensor, Norm



    if is_training:
        mode = 'train'
        num_workers = configs.num_workers
    else:
        mode = 'test'
        num_workers = 0
    dataset = data_set(
        configs=configs,
        data_train_path=data_train_path,
        data_test_path=data_test_path,
        mode=mode,
        transform=transforms.Compose([Norm(), ToTensor()]))
    return DataLoader(dataset,
                      pin_memory=True,
                      batch_size=batch_size,
                      shuffle=is_shuffle,
                      num_workers=num_workers)

