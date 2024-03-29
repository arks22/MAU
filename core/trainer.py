import os
import cv2
import math
import torch
import json
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess
import skvideo.io
import lpips


def train(model, ims, real_input_flag, configs, itr):
    _, loss_mse = model.train(ims, real_input_flag, itr)
    return np.round(loss_mse, 6)


def test(model, input_handle, configs, epoch, timestamp, is_valid):
    gen_path = os.path.join(configs.gen_frm_dir, timestamp)

    num_val = len(input_handle)
    if is_valid:
        print('\nValid with', num_val, 'data')
        print('Save samples:', configs.num_save_samples)

        res_path = os.path.join(gen_path, str(epoch))

    else:
        print('\nTest with', num_val, 'data')

        res_path = os.path.join(gen_path, 'test')


    loss_fn_lpips = lpips.LPIPS(net='alex', spatial=True).to(configs.device)

    
    if not is_valid or (not os.path.exists(res_path) and epoch % configs.sample_interval == 0):
        os.mkdir(res_path)
        os.chmod(res_path,0o777)
        os.mkdir(os.path.join(res_path, 'movie'))
        os.chmod(os.path.join(res_path, 'movie'), 0o777)
        os.mkdir(os.path.join(res_path, 'image'))
        os.chmod(os.path.join(res_path, 'image'), 0o777)
        os.mkdir(os.path.join(res_path, 'ndarray'))
        os.chmod(os.path.join(res_path, 'ndarray'), 0o777)

    batch_id = 0
    output_length = min(configs.total_length - configs.input_length, configs.total_length - 1)
    img_mse   = np.zeros(output_length)
    img_psnr  = np.zeros(output_length)
    img_ssim  = np.zeros(output_length)
    img_lpips = np.zeros(output_length)
    scores = []

    for data in input_handle:
        if num_val < batch_id: break;
        print('\ritr:' + str(batch_id + 1), end='')
        batch_scores = []

        batch_id += 1
        batch_size = data.shape[0]
        real_input_flag = np.zeros(
            (batch_size,
            configs.total_length - configs.input_length - 1,
            configs.img_height // configs.patch_size,
            configs.img_width // configs.patch_size,
            configs.patch_size ** 2 * configs.in_channel))

        img_gen = model.test(data, real_input_flag)
        img_gen = img_gen.transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
        test_ims = data.detach().cpu().numpy().transpose(0, 1, 3, 4, 2)  # * 0.5 + 0.5
        test_ims = preprocess.reshape_patch_back(test_ims, configs.patch_size)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        img_out = img_gen[:, -output_length:, :]

        for i in range(output_length):
            gt = test_ims[:, i + configs.input_length, :]

            gx = img_out[:, i, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            
            # MSEの計算
            mse_score = np.square(gt - gx).mean() / batch_size


            # LPIPSの計算
            t1 = torch.from_numpy((gt - 0.5) / 0.5).to(configs.device)
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

            # PSNRの計算
            psnr_score = 0
            for sample_id in range(batch_size):
                mse_tmp = np.square(gt[sample_id, :] - gx[sample_id, :]).mean()
                psnr_score += 10 * np.log10(1 / mse_tmp)
            psnr_score /= (batch_size)

            # SSIMの計算
            ssim_score = 0
            for b in range(batch_size):
                ssim_score += compare_ssim(gt[b, :], gx[b, :], channel_axis = -1, data_range=1)
            ssim_score /= batch_size

            img_mse[i] += mse_score
            img_psnr[i] += psnr_score
            img_ssim[i] += ssim_score
            img_lpips[i] += lpips_score
            
            batch_scores.append({ "mse": mse_score.item(), "psnr": psnr_score.item(), "ssim": ssim_score.item(), "lpips": lpips_score.item() })

        scores.append(batch_scores)


        # validの時はnum_save_samplesの数保存、テストの時は全て
        if (is_valid  and epoch % configs.sample_interval == 0 and batch_id <= configs.num_save_samples) or not is_valid:
            res_width = configs.img_width
            res_height = configs.img_height
            img = np.ones((2 * res_height, configs.total_length * res_width, configs.in_channel))

            video_data = np.zeros((res_height, res_width*2, configs.in_channel, configs.total_length))
            video_name = os.path.join(res_path, 'movie', str(batch_id).zfill(2) + '.mp4')
            writer = skvideo.io.FFmpegWriter(video_name, inputdict={'-r':'1'}, outputdict={'-r':'1','-pix_fmt':'yuv420p','-vcodec': 'libx264'})

            # make video and image for GT
            for i in range(configs.total_length):
                img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :, :, :]
                video_data[:res_height, :res_width, :, i] = test_ims[0, i, :, :, :]

            # make video and image for predicted
            for i in range(output_length):
                img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width, :] = img_out[0, -output_length + i, :]
                video_data[:res_height, res_width:, : ,configs.input_length + i] = img_out[0, -output_length + i, :, :, :]
            np.save(os.path.join(res_path, 'ndarray', str(batch_id).zfill(2)), img_out[0])

            # release video
            video_data = np.maximum(video_data, 0)
            video_data = np.minimum(video_data, 1)
            video_data = (video_data * 255).astype(np.uint8)

            for i in range(configs.total_length):
                frame = video_data[:,:,:,i].astype(np.uint8)

                if configs.in_channel == 1:
                    frame = np.repeat(frame, 3).reshape(res_height, res_width*2, 3)

                writer.writeFrame(frame)
            writer.close()

            img = np.maximum(img, 0)
            img = np.minimum(img, 1)
            img_name = os.path.join(res_path, 'image', str(batch_id).zfill(2) + '.png')
            cv2.imwrite(img_name, (img * 255).astype(np.uint8))

    img_mse   /= num_val
    img_psnr  /= num_val
    img_ssim  /= num_val
    img_lpips /= num_val
    avg_mse   = np.mean(img_mse).item()
    avg_psnr  = np.mean(img_psnr).item()
    avg_ssim  = np.mean(img_ssim).item()
    avg_lpips = np.mean(img_lpips).item()

    summary = {
        "mse_avg": avg_mse,
        "psnr_avg": avg_psnr,
        "ssim_avg": avg_ssim,
        "lpips_avg": avg_lpips,
        "mse": img_mse.tolist(),
        "psnr": img_psnr.tolist(),
        "ssim": img_ssim.tolist(),
        "lpips": img_lpips.tolist()
    }
    epoch_result = { "scores": scores, "summary": summary }

    with open(os.path.join(gen_path, 'results.json'), 'r') as f:
        result_json = json.load(f)
        if is_valid:
            result_json['valid'].append(epoch_result)
        else:
            result_json['test'] = epoch_result
    with open(os.path.join(gen_path, 'results.json'), 'w') as f:
        json.dump(result_json, f, indent=4)

    print()
    print('------------------------------------------------------------------------------------------')
    for i in range(output_length):
        print('|   ' + str(i+1) + '   ', end='')
    print('|')


    print('| -- *MSE  per frame: ' + str(avg_mse) + ' ---------------------------------------------------')
    for i in range(output_length):
        print('| ' + str(img_mse[i])[:5] + ' ', end='')
    print('|')

    print('| --  *PSNR  per frame: ' + str(avg_psnr) + ' -----------------------------------------------')
    for i in range(output_length):
        print('| ' + str(img_psnr[i])[:5] + ' ', end='')
    print('|')

    print('| -- *SSIM per frame: ' + str(avg_ssim) + ' -------------------------------------------------')
    for i in range(output_length):
        print('| ' + str(img_ssim[i])[:5] + ' ', end='')
    print('|')

    print('| -- *LPIPS per frame: ' + str(avg_lpips) + ' --------------------------------------------------')
    for i in range(output_length):
        print('| ' + str(img_lpips[i])[:5] + ' ', end='')
    print('|')
    print('----------------------------------------------------------------------------------------')
