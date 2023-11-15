import os
import numpy as np
from core.data_provider import datasets_factory
from core.models.model_factory import Model
import core.trainer as trainer
import pynvml
import time
import datetime
from tqdm import tqdm
import json

from run_utils.plot_loss import plot_loss
from run_utils.set_args import set_args

# !example
# python3 run.py --config=aia211

TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d%H%M')

pynvml.nvmlInit()

def schedule_sampling(args, eta, itr):
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


# トレーニング
def train_wrapper(args, model):
    begin = 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    #meminfo_begin = pynvml.nvmlDeviceGetMemoryInfo(handle)

    if args.pretrained_model:
        model.load(args.pretrained_model)
        begin = int(args.pretrained_model.split('-')[-1])

    train_input_handle = datasets_factory.data_provider(configs=args, dataset=args.dataset, path=args.data_train_path, batch_size=args.batch_size, mode = 'train', is_shuffle=True)
    val_input_handle = datasets_factory.data_provider(  configs=args, dataset=args.dataset, path=args.data_val_path,   batch_size=1, mode = 'val', is_shuffle=False)
    test_input_handle = datasets_factory.data_provider( configs=args, dataset=args.dataset, path=args.data_test_path,  batch_size=1, mode = 'test', is_shuffle=False)

    eta = args.sampling_start_value
    eta -= (begin * args.sampling_changing_rate)
    itr = begin
    time_train_start = time.time() 
    train_size = len(train_input_handle)

    for epoch in range(1, args.max_epoches + 1):
        print(f"------------- epoch: {epoch} / {args.max_epoches} ----------------")
        print(f"Train with {train_size}  batch")
        time_epoch_start = time.time() 

        with tqdm(total=train_size, desc="Train", leave=False) as pbar:
            for ims in train_input_handle:
                #if itr > 3: break ############ DEBUG ##############
                time_itr_start = time.time() 
                batch_size = ims.shape[0]
                eta, real_input_flag = schedule_sampling(args, eta, itr)
                loss = list(trainer.train(model, ims, real_input_flag, args, itr))

                pbar.set_postfix({"L2 Loss": loss[1]})
                pbar.update()
                itr += 1
                
        trainer.test(model, val_input_handle, args, epoch, TIMESTAMP, True)

        with open(os.path.join(gen_path, 'results.json'), 'r') as f:
            result_json = json.load(f)
            result_json['valid'][epoch-1]['summary']['l1loss'] = loss[0].item()
            result_json['valid'][epoch-1]['summary']['l2loss'] = loss[1].item()

        with open(os.path.join(gen_path, 'results.json'), 'w') as f:
            json.dump(result_json, f, indent=4)

        plot_loss(args, TIMESTAMP, train_size)
        model.save(TIMESTAMP, itr)
        time_epoch = round((time.time() - time_epoch_start) / 60, 3)
        pred_finish_time = time_epoch * (args.max_epoches - epoch) / 60
        print(f'{time_epoch} m/epoch | ETA: {round(pred_finish_time,2)} h')

    train_finish_time = round((time.time() - time_train_start) / 3600,2)

    trainer.test(model, test_input_handle, args, epoch, TIMESTAMP, False) #学習回し終わった後にテスト
    plot_loss(args, TIMESTAMP, train_size, train_finish_time)


# テスト
def test_pretrained(args, model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(configs=args, dataset=args.dataset, path=args.data_test_path, batch_size=1, mode = 'test', is_shuffle=False)
    trainer.test(model, test_input_handle, args, 1, TIMESTAMP, False)


if __name__ == '__main__':
    args = set_args()

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
        test_pretrained(args, model)
    else:
        save_path = os.path.join(args.save_dir, TIMESTAMP)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.chmod(save_path, 0o777)
        print('save results : ' + str(TIMESTAMP))
        train_wrapper(args, model)