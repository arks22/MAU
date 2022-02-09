import os
import argparse
import numpy as np
from core.data_provider import datasets_factory
from core.models.model_factory import Model
import core.trainer as trainer
import glob
import pynvml

pynvml.nvmlInit()
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='MAU')
parser.add_argument('--dataset', type=str, default='kitti')
parser.add_argument('--config', type=str, default='')

args_main = parser.parse_args()
args_main.tied = True

if args_main.config == 'mnist':
    from configs.mnist_configs import configs
elif args_main.config == 'kitti':
    from configs.kitti_configs import configs
elif args_main.config == 'town':
    from configs.town_configs import configs
elif args_main.config == 'aia211_1':
    from configs.aia211_1_configs import configs
elif args_main.config == 'aia211_5':
    from configs.aia211_5_configs import configs

parser = configs()
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
args.tied = True


def schedule_sampling(eta, itr, channel, batch_size):
    zeros = np.zeros((batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))

    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    #print('eta: ', eta)
    random_flip = np.random.random_sample(
        (batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * channel))
    real_input_flag = []
    for i in range(batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * channel))
    return eta, real_input_flag


def train_wrapper(model):
    begin = 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if args.pretrained_model:
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])

    save_dirs = glob.glob(args.save_dir + '/*')
    dirs = list(map(lambda d: int(os.path.basename(d)), save_dirs))
    dirs.sort()
    last_dir = str(dirs[-1] + 1) if len(dirs) > 0 else "1"

    args.save_dir    = os.path.join(args.save_dir,    last_dir)
    args.gen_frm_dir = os.path.join(args.gen_frm_dir, last_dir)
    os.mkdir(args.save_dir)
    os.mkdir(args.gen_frm_dir)

    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=args.data_train_path,
                                                        dataset=args.dataset,
                                                        data_test_path=args.data_val_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)

    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=args.data_train_path,
                                                      dataset=args.dataset,
                                                      data_test_path=args.data_val_path,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False)

    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    # real_input_flag = {}

    #Train
    for epoch in range(0, args.max_epoches):
        print("------------- epoch: " + str(epoch) + " / " + str(args.max_epoches) + " ----------------")
        print("train with " + str(len(train_input_handle)) + " data")

        for ims in train_input_handle:
            batch_size = ims.shape[0]
            eta, real_input_flag = schedule_sampling(eta, itr, args.img_channel, batch_size)
            trainer.train(model, ims, real_input_flag, args, itr)
            meminfo_end = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print("\rGPU memory:%dM | " % ((meminfo_end.used - meminfo_begin.used) / (1024 ** 2)),end='')

            itr += 1

        print('\nValidate:')
        trainer.test(model, val_input_handle, args, itr)
        model.save(itr)


def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       data_train_path=args.data_train_path,
                                                       dataset=args.dataset,
                                                       data_test_path=args.data_test_path,
                                                       batch_size=args.batch_size,
                                                       is_training=False,
                                                       is_shuffle=False)

    itr = 1
    for i in range(itr):
        trainer.test(model, test_input_handle, args, itr)


if __name__ == '__main__':

    print('Initializing models')
    if args.is_training == 'True':
        args.is_training = True
    else:
        args.is_training = False

    model = Model(args)

    if args.is_training:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        train_wrapper(model)
    else:
        if not os.path.exists(args.gen_frm_dir):
            os.makedirs(args.gen_frm_dir)
        test_wrapper(model)
