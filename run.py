import os
import argparse
import importlib
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
import json
import datetime

# !example
# python3 run.py --config=aia211

dt_now = datetime.datetime.now()
TIMESTAMP = dt_now.strftime('%Y%m%d%H%M')

pynvml.nvmlInit()
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='MAU')
parser.add_argument('--config',  type=str, required=True)
parser.add_argument('--test',                   action='store_true')
parser.add_argument('--dataset',                type=str)
parser.add_argument('--data_train_path',        type=str)
parser.add_argument('--data_val_path',          type=str)
parser.add_argument('--data_test_path',         type=str)
parser.add_argument('--input_length',           type=int)
parser.add_argument('--real_length',            type=int)
parser.add_argument('--total_length',           type=int)
parser.add_argument('--img_height',             type=int)
parser.add_argument('--img_width',              type=int)
parser.add_argument('--sr_size',                type=int, help='Encoder-Decoderモデルの層の深さを決定')
parser.add_argument('--in_channel',             type=int, help='入力チャンネル数')
parser.add_argument('--out_channel',            type=int)
parser.add_argument('--patch_size',             type=int)
parser.add_argument('--alpha',                  type=float )
parser.add_argument('--model_name',             type=str)   
parser.add_argument('--num_workers',            type=int)   
parser.add_argument('--num_hidden',             type=int)   
parser.add_argument('--num_layers',             type=int, help='MAUセルの深さ(スタック数)を決定')   
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
parser.add_argument('--sample_interval',        type=int)   
parser.add_argument('--num_save_samples',       type=int)   
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

configs = importlib.import_module(f"configs.{args.config}").configs
args = configs(args)

if args.test:
    mode = 'Test'
    dataset_path = args.data_test_path
else:
    mode = 'Train'
    dataset_path = args.data_train_path

print('---------------------------------------------')
print('Dataset type  :', args.dataset)
print('Mode          :', mode)
print('Dataset path  :', dataset_path)
print('Configuration :', args.config)
print('Model         :', args.model_name)
print('---------------------------------------------')


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.in_channel))

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
                    args.patch_size ** 2 * args.in_channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.in_channel))
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
                                  args.patch_size ** 2 * args.in_channel))
    return eta, real_input_flag


def plot_loss(finish_time=0):
    l1loss = []
    l2loss = []
    mse = []
    psnr = []
    ssim = []
    lpips = []
    with open(os.path.join(gen_path, 'results.json'), 'r') as f:
        valid_results = json.load(f)['valid']
        for result in valid_results:
            l1loss.append(result['summary']['l1loss'])
            l2loss.append(result['summary']['l2loss'])
            mse.append(result['summary']['mse_avg'])
            psnr.append(result['summary']['psnr_avg'])
            ssim.append(result['summary']['ssim_avg'])
            lpips.append(result['summary']['lpips_avg'])

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3,3)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0:2], fc='gray', xlim=(0,10),ylim=(0,10)))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[2, 0]))
    ax.append(fig.add_subplot(gs[1, 1]))
    ax.append(fig.add_subplot(gs[2, 1]))
    ax.append(fig.add_subplot(gs[1, 2]))
    ax.append(fig.add_subplot(gs[2, 2]))

    ax[1].plot(l1loss, color='r', lw=0.75, label='train loss')
    ax[2].plot(l2loss, color='b', lw=0.75, label='train L2_loss')
    ax[3].plot(mse, color='g', lw=0.75, label='valid mse')
    ax[4].plot(psnr, color='y', lw=0.75, label='valid psnr')
    ax[5].plot(ssim, color='m', lw=0.75, label='valid ssim')
    ax[6].plot(lpips, color='c', lw=0.75, label='valid lpips')

    for x in ax:
        x.grid()
        x.legend()

    ax[0].xaxis.set_major_locator(mpl.ticker.NullLocator())
    ax[0].yaxis.set_major_locator(mpl.ticker.NullLocator())
    ax[0].text(1,9,"MAU " + str(TIMESTAMP))
    ax[0].text(1,8,"---------------------")
    ax[0].text(1,7,"Epochs: " + str(args.max_epoches))
    ax[0].text(1,6,'Dataset ' + str(args.dataset) + ' with size ' +  str(train_size) + ' * ' + str(args.batch_size)) 
    ax[0].text(1,5,"Resolution: " + str(args.img_height) + ' * ' + str(args.img_width))
    if not finish_time == 0:
        ax[0].text(1,3,"Time: " + str(finish_time) + 'h')

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

    train_input_handle = datasets_factory.data_provider(configs=args,
                                                        dataset=args.dataset,
                                                        path=args.data_train_path,
                                                        batch_size=args.batch_size,
                                                        mode = 'train',
                                                        is_shuffle=True)
    val_input_handle = datasets_factory.data_provider(configs=args,
                                                      dataset=args.dataset,
                                                      path=args.data_val_path,
                                                      batch_size=1,
                                                      mode = 'val',
                                                      is_shuffle=False)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                      dataset=args.dataset,
                                                      path=args.data_test_path,
                                                      batch_size=1,
                                                      mode = 'test',
                                                      is_shuffle=False)

    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    time_train_start = time.time() 
    global train_size
    train_size = len(train_input_handle)

    for epoch in range(1, args.max_epoches + 1):
        print("------------- epoch: " + str(epoch) + " / " + str(args.max_epoches) + " ----------------")
        print("Train with " + str(train_size) + " batch")
        time_epoch_start = time.time() 

        for ims in train_input_handle:
            #if itr > 3: break ############ DEBUG ##############
            time_itr_start = time.time() 
            batch_size = ims.shape[0]
            eta, real_input_flag = schedule_sampling(eta, itr)
            loss = list(trainer.train(model, ims, real_input_flag, args, itr))

            time_itr = round(time.time() - time_itr_start, 3)
            print('\ritr:', itr - ((epoch - 1) * train_size), str(time_itr).ljust(5,'0'), 's | L1 loss:', loss[0], 'L2 loss:', loss[1], end='')
            itr += 1

        trainer.test(model, val_input_handle, args, epoch, TIMESTAMP, True)

        with open(os.path.join(gen_path, 'results.json'), 'r') as f:
            result_json = json.load(f)
            result_json['valid'][epoch-1]['summary']['l1loss'] = loss[0].item()
            result_json['valid'][epoch-1]['summary']['l2loss'] = loss[1].item()
        with open(os.path.join(gen_path, 'results.json'), 'w') as f:
            json.dump(result_json, f, indent=4)

        plot_loss() 
        model.save(TIMESTAMP,itr)
        time_epoch = round((time.time() - time_epoch_start) / 60, 3)
        pred_finish_time = time_epoch * (args.max_epoches - epoch) / 60
        print(str(time_epoch) + 'm/epoch | ETA: ' + str(round(pred_finish_time,2)) + 'h')

    train_finish_time = round((time.time() - time_train_start) / 3600,2)

    trainer.test(model, test_input_handle, args, epoch, TIMESTAMP, False) #学習回し終わった後にテスト
    plot_loss(train_finish_time)


def test_pretrained(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args,
                                                       dataset=args.dataset,
                                                       path=args.data_test_path,
                                                       batch_size=args.batch_size,
                                                       mode = 'test',
                                                       is_shuffle=False)

    trainer.test(model, test_input_handle, args, 1, TIMESTAMP, False)


if __name__ == '__main__':
    print('Initializing models')
    model = Model(args)

    gen_path = os.path.join(args.gen_frm_dir, TIMESTAMP)
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)
        os.chmod(gen_path, 0o777)

    config_json_path = os.path.join(gen_path,'configs.json')
    with open(config_json_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    result_json_path = os.path.join(gen_path,'results.json')
    init_dict = {'test':{}, 'valid':[]}
    with open(result_json_path, 'w') as f:
        json.dump(init_dict, f, indent=4)

    if args.test:
        test_pretrained(model)
    else:
        save_path = os.path.join(args.save_dir, TIMESTAMP)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.chmod(save_path, 0o777)
        print('save results : ' + str(TIMESTAMP))
        train_wrapper(model)
