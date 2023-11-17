import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import os

def plot_loss(args, timestamp, train_size, finish_time=0):
    train_loss, valid_loss, psnr, ssim, lpips = [], [], [], [], []

    gen_path = os.path.join(args.gen_frm_dir, timestamp)

    with open(os.path.join(gen_path, 'results.json'), 'r') as f:
        valid_results = json.load(f)['valid']
        for result in valid_results:
            train_loss.append(result['summary']['train_loss'])
            valid_loss.append(result['summary']['mse_avg'])
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

    ax[1].plot(train_loss, color='b', lw=0.75, label='train loss (MSE)')
    ax[2].plot(valid_loss, color='g', lw=0.75, label='valid loss (MSE)')
    ax[3].plot(psnr, color='y', lw=0.75, label='valid psnr')
    ax[4].plot(ssim, color='m', lw=0.75, label='valid ssim')
    ax[5].plot(lpips, color='c', lw=0.75, label='valid lpips')

    for x in ax:
        x.grid()
        x.legend()

    ax[0].xaxis.set_major_locator(mpl.ticker.NullLocator())
    ax[0].yaxis.set_major_locator(mpl.ticker.NullLocator())
    ax[0].text(1,9, f'MAU {timestamp}')
    ax[0].text(1,8, '---------------------')
    ax[0].text(1,7, f'Epochs:  {args.max_epoches}')
    ax[0].text(1,6, f'Dataset {args.data_train_path}  with * {args.batch_size} batch * {train_size}')
    ax[0].text(1,5, f'Image : {args.img_height} px * {args.img_width} px * {args.in_channel} ch')
    ax[0].text(1,3, f'lr: {args.lr}, lr_decay: {args.lr_decay}, decay_interval: {args.delay_interval}, decay start at: {args.sampling_stop_iter}')
    ax[0].text(1,2, f'Resolution: {args.img_height}  * {args.img_width}')

    if not finish_time == 0:
        ax[0].text(1,1, f'Time: {finish_time} h')

    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig_path = os.path.join(args.gen_frm_dir, timestamp, 'losses.png')
    fig.savefig(fig_path, format="png", dpi=200)
