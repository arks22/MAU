import os
import argparse
import numpy as np
from core.data_provider import datasets_factory
from core.models.model_factory import Model
import core.trainer as trainer
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import pynvml
import time
import datetime

dt_now = datetime.datetime.now()
TIMESTAMP = dt_now.strftime('%Y%m%d%H%M')

pynvml.nvmlInit()
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='MAU')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--is_training',            type=bool)
parser.add_argument('--data_train_path',        type=str)
parser.add_argument('--data_test_path',         type=str)
parser.add_argument('--input_length',           type=int)
parser.add_argument('--real_length',            type=int)
parser.add_argument('--total_length',           type=int)
parser.add_argument('--img_height',             type=int)
parser.add_argument('--img_width',              type=int)
parser.add_argument('--sr_size',                type=int)
parser.add_argument('--img_channel',            type=int)
parser.add_argument('--patch_size',             type=int)
parser.add_argument('--alpha',                  type=float )
parser.add_argument('--model_name',             type=str)   
parser.add_argument('--num_workers',            type=int)   
parser.add_argument('--num_hidden',             type=int)   
parser.add_argument('--num_layers',             type=int)   
parser.add_argument('--num_heads',              type=int)   
parser.add_argument('--filter_size',            type=int)   
parser.add_argument('--stride',                 type=int)   
parser.add_argument('--time',                   type=int)   
parser.add_argument('--time_stride',            type=int)   
parser.add_argument('--tau',                    type=int)   
parser.add_argument('--cell_mode',              type=str)   
parser.add_argument('--model_mode',             type=str)  
parser.add_argument('--lr',                     type=float) 
parser.add_argument('--lr_decay',               type=float) 
parser.add_argument('--delay_interval',         type=float) 
parser.add_argument('--batch_size',             type=int)   
parser.add_argument('--max_epoches',            type=int)   
parser.add_argument('--num_save_samples',       type=int)   
parser.add_argument('--num_valid_samples',       type=int)   
parser.add_argument('--n_gpu',                  type=int)   
parser.add_argument('--device',                 type=str)   
parser.add_argument('--pretrained_model',       type=str)
parser.add_argument('--save_dir',               type=str)
parser.add_argument('--gen_frm_dir',            type=str)
parser.add_argument('--scheduled_sampling',     type=bool)  
parser.add_argument('--sampling_stop_iter',     type=int)  
parser.add_argument('--sampling_start_value',   type=float) 
parser.add_argument('--sampling_changing_rate', type=float) 
args = parser.parse_args()

if args.config == 'mnist':
    from configs.mnist_configs import configs
elif args.config == 'kitti':
    from configs.kitti_configs import configs
elif args.config == 'town':
    from configs.town_configs import configs
elif args.config == 'aia211':
    from configs.aia211_configs import configs
args = configs(args)

print('---------------------------------------------')
print('Dataset       :', args.dataset)
print('Configuration :', args.config)
print('---------------------------------------------')


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0

    random_flip = np.random.random_sample((args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_height // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def plot_loss(indices, finish_time):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3,3)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0:2],fc='gray', xlim=(0,10),ylim=(0,10)))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[2, 0]))
    ax.append(fig.add_subplot(gs[1, 1]))
    ax.append(fig.add_subplot(gs[2, 1]))
    ax.append(fig.add_subplot(gs[1, 2]))
    ax.append(fig.add_subplot(gs[2, 2]))

    ax[1].plot(indices[0].flatten(), color='r', lw=0.75, label='train loss')
    ax[2].plot(indices[1].flatten(), color='b', lw=0.75, label='train L2_loss')
    ax[3].plot(indices[2].flatten(), color='g', lw=0.75, label='valid mse')
    ax[4].plot(indices[3].flatten(), color='y', lw=0.75, label='valid psnr')
    ax[5].plot(indices[4].flatten(), color='m', lw=0.75, label='valid ssim')
    ax[6].plot(indices[5].flatten(), color='c', lw=0.75, label='valid lpips')

    for i in range(len(ax)):
        ax[i].grid()
        ax[i].legend()

    ax0.xaxis.set_major_locator(mpl.ticker.NullLocator())
    ax0.yaxis.set_major_locator(mpl.ticker.NullLocator())
    ax0.text(1,9,"MAU " + str(TIMESTAMP))
    ax0.text(1,8,"---------------------")
    ax0.text(1,7,"Dataset " + str(args.dataset))
    ax0.text(1,6,"Batch size: " + str(args.batch_size))
    ax0.text(1,5,"Epoch: " + str(args.max_epoches))
    time = '- ' if finish_time == 0 else str(finish_time)
    ax0.text(1,4,"Time: " + time + 'h')

    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig_path = os.path.join(args.gen_frm_dir, TIMESTAMP, 'losses.png')
    fig.savefig(fig_path, format="png", dpi=200)


def train_wrapper(model):
    begin = 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if args.pretrained_model:
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])

#    save_dirs = glob.glob(args.save_dir + '/*')
#    dirs = list(map(lambda d: int(os.path.basename(d)), save_dirs))
#    dirs.sort()
#    last_dir = str(dirs[-1] + 1) if len(dirs) > 0 else "1"

#    args.save_dir    = os.path.join(args.save_dir,    last_dir)
#    args.gen_frm_dir = os.path.join(args.gen_frm_dir, last_dir)
#    os.mkdir(args.save_dir)
#    os.mkdir(args.gen_frm_dir)

    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        data_train_path=args.data_train_path,
                                                        dataset=args.dataset,
                                                        data_test_path=args.data_test_path,
                                                        batch_size=args.batch_size,
                                                        is_training=True,
                                                        is_shuffle=True)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                      data_train_path=args.data_train_path,
                                                      dataset=args.dataset,
                                                      data_test_path=args.data_test_path,
                                                      batch_size=args.batch_size,
                                                      is_training=False,
                                                      is_shuffle=False)

    print(train_input_handle)
    print(test_input_handle)
    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    indices = np.zeros((6,args.max_epoches))
    time_train_start = time.time() 

    for epoch in range(1, args.max_epoches + 1):
        print("------------- epoch: " + str(epoch) + " / " + str(args.max_epoches) + " ----------------")
        print("Train with " + str(len(train_input_handle)) + " data")
        time_epoch_start = time.time() 

        for ims in train_input_handle:
            time_itr_start = time.time() 
            batch_size = ims.shape[0]
            eta, real_input_flag = schedule_sampling(eta, itr)
            loss = trainer.train(model, ims, real_input_flag, args, itr)

            time_itr = round(time.time() - time_itr_start, 3)
            print('\ritr:' + str(itr) + ' ' + str(time_itr).ljust(5,'0') + 's | L1 loss: ' + str(loss[0]) + ' L2 loss: ' + str(loss[1]) , end='')
            itr += 1

        test_indices = trainer.test(model, test_input_handle, args, itr, TIMESTAMP, True)
        indices[:, epoch-1] = loss + test_indices
        plot_loss(indices,0)
        model.save(TIMESTAMP,itr)
        time_epoch = round((time.time() - time_epoch_start) / 60, 3)
        pred_finish_time = time_epoch * (args.max_epoches - epoch) / 60
        print(str(time_epoch) + 'm/epoch | ETA: ' + str(round(pred_finish_time,2)) + 'h')

    train_finish_time = round((time.time() - time_train_start) / 3600,2)
    trainer.test(model, test_input_handle, args, itr, TIMESTAMP, False)
    plot_loss(indices,train_finish_time)


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
        trainer.test(model, test_input_handle, args, itr, TIMESTAMP, False)


if __name__ == '__main__':
    print(args.model_name)
    print('Initializing models')
    model = Model(args)

    gen_path = os.path.join(args.gen_frm_dir, TIMESTAMP)
    if not os.path.exists(gen_path): os.mkdir(gen_path)

    if args.is_training:
        save_path = os.path.join(args.save_dir, TIMESTAMP)
        if not os.path.exists(save_path): os.mkdir(save_path)
        print('save results : ' + str(TIMESTAMP))
        train_wrapper(model)
    else:
        test_wrapper(model)
