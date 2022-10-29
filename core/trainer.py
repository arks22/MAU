import os
import cv2
import math
import torch
import codecs
import lpips
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess


def train(model, ims, real_input_flag, configs, itr):
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    return np.round(loss_l1, 6), np.round(loss_l2, 6)


def test(model, test_input_handle, configs, epoch, timestamp):
    num_val = min(len(test_input_handle), configs.num_val_samples)
    print('\nTest with', num_val, 'data')

    loss_fn_lpips = lpips.LPIPS(net='alex', spatial=True).to(configs.device)
    res_path = os.path.join(configs.gen_frm_dir, timestamp, str(epoch))
    if not os.path.exists(res_path): os.mkdir(res_path)

    batch_id = 0
    img_mse, img_psnr, img_ssim, img_lpips = [], [], [], []

    output_length = min(configs.total_length - configs.input_length, configs.total_length - 1)
    img_mse   = np.zeros(output_length)
    img_psnr  = np.zeros(output_length)
    img_ssim  = np.zeros(output_length)
    img_lpips = np.zeros(output_length)

    for data in test_input_handle:
        if num_val < batch_id: break;
        print('\ritr:' + str(batch_id + 1),end='')

        batch_id += 1
        batch_size = data.shape[0]
        real_input_flag = np.zeros(
            (batch_size,
            configs.total_length - configs.input_length - 1,
            configs.img_height // configs.patch_size,
            configs.img_width // configs.patch_size,
            configs.patch_size ** 2 * configs.img_channel))

        img_gen = model.test(data, real_input_flag)
        img_gen = img_gen.transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
        test_ims = data.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
        test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        img_out = img_gen[:, -output_length:, :]

        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :]
            gx = img_out[:, i, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse_score = np.square(x - gx).sum() / batch_size

            t1 = torch.from_numpy((x - 0.5) / 0.5).to(configs.device)
            t1 = t1.permute((0, 3, 1, 2))
            t2 = torch.from_numpy((gx - 0.5) / 0.5).to(configs.device)
            t2 = t2.permute((0, 3, 1, 2))
            shape = t1.shape
            if not shape[1] == 3:
                new_shape = (shape[0], 3, *shape[2:])
                t1.expand(new_shape)
                t2.expand(new_shape)
            d = loss_fn_lpips.forward(t1, t2)
            lpips_score = d.mean()
            lpips_score = lpips_score.detach().cpu().numpy()

            psnr_score = 0
            for sample_id in range(batch_size):
                mse_tmp = np.square(x[sample_id, :] - gx[sample_id, :]).mean()
                psnr_score += 10 * np.log10(1 / mse_tmp)
            psnr_score /= (batch_size)

            ssim_score = 0
            for b in range(batch_size):
                ssim_score += compare_ssim(x[b, :], gx[b, :], channel_axis = -1)
            ssim_score /= batch_size

            img_mse[i] += mse_score
            img_psnr[i] += psnr_score
            img_ssim[i] += ssim_score
            img_lpips[i] += lpips_score


        if batch_id <= configs.num_save_samples:
            #関数化したい
            res_width = configs.img_width
            res_height = configs.img_height
            img = np.ones((2 * res_height, configs.total_length * res_width, configs.img_channel))
            img_name = os.path.join(res_path, str(batch_id) + '.png')

            vid_arr = np.ones((res_height, res_width*2, configs.img_channel, configs.total_length))
            vid_name = os.path.join(res_path, str(batch_id) + '.mp4')
            fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
            video  = cv2.VideoWriter(vid_name, fourcc, 1.00, (res_width*2+1, res_height+1))
            #なぜかFPS2.00が選べない。なぜ？
            for i in range(configs.total_length):
                img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :,:,:]
                vid_arr[:res_height, :res_width, :, i] = test_ims[0, i, :,:,:]

            for i in range(output_length):
                img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width, :] = img_out[0, -output_length + i, :]
                vid_arr[:res_height, res_width:, : ,configs.input_length + i] = img_out[0, -output_length + i, :, :, :]

            for i in range(configs.total_length):
                frame = vid_arr[:,:,:,i]
                frame = np.maximum(frame, 0)
                frame = np.minimum(frame, 1)
                cv2.imwrite('tmp.png', (frame * 255).astype(np.uint8))
                tmp = cv2.imread('tmp.png')
                video.write(tmp)
            os.remove('tmp.png')
            video.release()

            img = np.maximum(img, 0)
            img = np.minimum(img, 1)
            cv2.imwrite(img_name, (img * 255).astype(np.uint8))

    img_mse   /= num_val
    img_psnr  /= num_val
    img_ssim  /= num_val
    img_lpips /= num_val
    avg_mse   = sum(img_mse)   / output_length
    avg_psnr  = sum(img_psnr)  / output_length
    avg_ssim  = sum(img_ssim)  / output_length
    avg_lpips = sum(img_lpips) / output_length

    print('\n')
    with codecs.open(res_path + '/data.txt', 'w+') as data_write:
        data_write.truncate()
        
        print('----------------------------------------------------------------------------------------------------')
        print('|    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |    9    |    10   |')

        print('| -- *MSE  per frame: ' + str(avg_mse) + ' --------------------------------------------------------')
        for i in range(output_length):
            print('| ' + str(img_mse[i])[:7] + ' ', end='')
        print('|')
        data_write.writelines(str(avg_mse) + '\n')
        data_write.writelines(str(img_mse) + '\n')

        print('| --  *PSNR  per frame: ' + str(avg_psnr) + ' -------------------------------------------------')
        for i in range(output_length):
            print('| ' + str(img_psnr[i])[:7] + ' ', end='')
        print('|')
        data_write.writelines(str(avg_psnr) + '\n')
        data_write.writelines(str(img_psnr) + '\n')

        print('| -- *SSIM per frame: ' + str(avg_ssim) + ' -------------------------------------------------')
        for i in range(output_length):
            print('| ' + str(img_ssim[i])[:7] + ' ', end='')
        print('|')
        data_write.writelines(str(avg_ssim) + '\n')
        data_write.writelines(str(img_ssim) + '\n')

        print('| -- *LPIPS per frame: ' + str(avg_lpips) + ' -----------------------------------------------')
        for i in range(output_length):
            print('| ' + str(img_lpips[i])[:7] + ' ', end='')
        print('|')
        data_write.writelines(str(avg_lpips) + '\n')
        data_write.writelines(str(img_lpips) + '\n')

        print('----------------------------------------------------------------------------------------------------')

    return avg_mse, avg_psnr, avg_ssim, avg_lpips
