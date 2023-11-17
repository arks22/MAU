import argparse

def configs(args):
    if args.model_name             == None : args.model_name              = 'mau'
    if args.dataset                == None : args.dataset                 = 'sun'
    #if args.data_train_path        == None : args.data_train_path         = 'data/aia211_tuned_3wave/aia211_tuned_3wave_train.npy' #Docker側でマウント
    #if args.data_val_path          == None : args.data_val_path           = 'data/aia211_tuned_3wave/aia211_tuned_3wave_val.npy'
    #if args.data_test_path         == None : args.data_test_path          = 'data/aia211_tuned_3wave/aia211_tuned_3wave_test.npy'
    if args.data_train_path        == None : args.data_train_path         = 'data/aia211_256/aia211_exp_train_mini.npy' #Docker側でマウント
    if args.data_val_path          == None : args.data_val_path           = 'data/aia211_256/aia211_exp_val_mini.npy' 
    if args.data_test_path         == None : args.data_test_path          = 'data/aia211_256/aia211_exp_test_mini.npy'
    if args.input_length           == None : args.input_length            = 12
    if args.real_length            == None : args.real_length             = 24
    if args.total_length           == None : args.total_length            = 24
    if args.img_height             == None : args.img_height              = 256
    if args.img_width              == None : args.img_width               = 256
    if args.sr_size                == None : args.sr_size                 = 4
    if args.patch_size             == None : args.patch_size              = 4
    if args.in_channel             == None : args.in_channel              = 1
    if args.out_channel            == None : args.out_channel             = 1
    if args.alpha                  == None : args.alpha                   = 1
    if args.num_workers            == None : args.num_workers             = 2
    if args.num_hidden             == None : args.num_hidden              = 64
    if args.num_layers             == None : args.num_layers              = 16
    if args.num_heads              == None : args.num_heads               = 4
    if args.filter_size            == None : args.filter_size             = (5, 5)
    if args.stride                 == None : args.stride                  = 1
    if args.time                   == None : args.time                    = 2
    if args.time_stride            == None : args.time_stride             = 1
    if args.tau                    == None : args.tau                     = 5
    if args.cell_mode              == None : args.cell_mode               = 'residual'
    if args.model_mode             == None : args.model_mode              = 'recall'
    if args.lr                     == None : args.lr                      = 5e-4
    if args.lr_decay               == None : args.lr_decay                = 0.97
    if args.delay_interval         == None : args.delay_interval          = 225# per 1 epocs
    if args.batch_size             == None : args.batch_size              = 4
    if args.max_epoches            == None : args.max_epoches             = 100
    if args.sample_interval        == None : args.sample_interval         = 5
    if args.num_save_samples       == None : args.num_save_samples        = 10
    if args.n_gpu                  == None : args.n_gpu                   = 1
    if args.device                 == None : args.device                  = 'cuda'
    if args.pretrained_model       == None : args.pretrained_model        = ''
    if args.save_dir               == None : args.save_dir                = 'checkpoints/aia211/'
    if args.gen_frm_dir            == None : args.gen_frm_dir             = 'results/aia211/'
    if args.scheduled_sampling     == None : args.scheduled_sampling      = True
    if args.sampling_stop_iter     == None : args.sampling_stop_iter      = 11250 #学習率を変更しない
    if args.sampling_start_value   == None : args.sampling_start_value    = 1.0
    if args.sampling_changing_rate == None : args.sampling_changing_rate  = 0.00005

    return args
