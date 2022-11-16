import os
import cv2
import math
import torch
import json
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess
import skvideo.io
#import lpips


def train(model, ims, real_input_flag, configs, itr):
    _, loss_l1, loss_l2 = model.train(ims, real_input_flag, itr)
    return np.round(loss_l1, 6), np.round(loss_l2, 6)


def test(model, test_input_handle, configs, epoch, timestamp, is_valid):
    num_val = min(len(test_input_handle), configs.num_val_samples)
    gen_path = os.path.join(configs.gen_frm_dir, timestamp)

    if is_valid:
        print('\nValid with', num_val, 'data')
        print('Save samples:', configs.num_save_samples)
    else:
        print('\nTest with', num_val, 'data')

    #loss_fn_lpips = lpips.LPIPS(net='alex', spatial=True).to(configs.device)

    res_path = os.path.join(gen_path, str(epoch))
    if not os.path.exists(res_path): os.mkdir(res_path)

    batch_id = 0
    output_length = min(configs.total_length - configs.input_length, configs.total_length - 1)
    img_mse   = np.zeros(output_length)
    img_psnr  = np.zeros(output_length)
    img_ssim  = np.zeros(output_length)
    #img_lpips = np.zeros(output_length)
    scores = []

    for data in test_input_handle:
        if is_valid and num_val < batch_id: break;
        print('\ritr:' + str(batch_id + 1), end='')
        batch_scores = []

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
            #d = loss_fn_lpips.forward(t1, t2)
            #lpips_score = d.mean()
            #lpips_score = lpips_score.detach().cpu().numpy()

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
            #img_lpips[i] += lpips_score
            
            #batch_scores.append({ "mse": mse_score.item(), "psnr": psnr_score.item(), "ssim": ssim_score.item(), "lpips": lpips_score.item() })
            batch_scores.append({ "mse": mse_score.item(), "psnr": psnr_score.item(), "ssim": ssim_score.item() })

        scores.append(batch_scores)


        # validの時はnum_save_samplesの数保存、テストの時は全て
        if (is_valid and batch_id <= configs.num_save_samples) or not is_valid:
            res_width = configs.img_width
            res_height = configs.img_height
            img = np.ones((2 * res_height, configs.total_length * res_width, configs.img_channel))

            video_data = np.zeros((res_height, res_width*2, configs.img_channel, configs.total_length))
            video_name = os.path.join(res_path, str(batch_id) + '.mp4')
            writer = skvideo.io.FFmpegWriter(video_name, inputdict={'-r':'1'}, outputdict={'-r':'1','-pix_fmt':'yuv420p','-vcodec': 'libx264'})

            # make video and image for GT
            for i in range(configs.total_length):
                img[:res_height, i * res_width:(i + 1) * res_width, :] = test_ims[0, i, :, :, :]
                video_data[:res_height, :res_width, :, i] = test_ims[0, i, :, :, :]

            print(test_ims[0, output_length:, :, :, :].shape)
            np.save('gt-' + str(batch_id), test_ims[0, output_length:, :, :, :])

            # make video and image for predicted
            for i in range(output_length):
                img[res_height:, (configs.input_length + i) * res_width:(configs.input_length + i + 1) * res_width, :] = img_out[0, -output_length + i, :]
                video_data[:res_height, res_width:, : ,configs.input_length + i] = img_out[0, -output_length + i, :, :, :]

            print(img_out[0, output_length:, :, :, :].shape)
            np.save('pred-' + str(batch_id), img_out[0, output_length:, :, :, :])

            # release video
            video_data = np.maximum(video_data, 0)
            video_data = np.minimum(video_data, 1)
            video_data = (video_data * 255).astype(np.uint8)
            for i in range(configs.total_length):
                frame = video_data[:,:,:,i].astype(np.uint8)
                color = (255, 255, 255)
                cv2.putText(frame, text = 't = ' + str(i+1), org=(20,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,  fontScale=0.8,  color=color,  thickness=1)
                if i >= configs.input_length:
                    txt_mse = 'MSE: ' + str(batch_scores[i-configs.input_length]['mse'])[:4]
                    txt_psnr = 'PSNR: ' + str(batch_scores[i-configs.input_length]['psnr'])[:4]
                    txt_ssim = 'SSIM: ' + str(batch_scores[i-configs.input_length]['ssim'])[:4]
                    #txt_lpips = 'LPIPS: ' + str(batch_scores[i-configs.input_length]['lpips'])[:4]
                    cv2.putText(frame, text=txt_mse, org=(517,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,  color=color,  thickness=1)
                    cv2.putText(frame, text=txt_psnr, org=(517,65), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,  color=color,  thickness=1)
                    cv2.putText(frame, text=txt_ssim, org=(517,80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,  color=color,  thickness=1)
                    #cv2.putText(frame, text=txt_lpips, org=(517,95), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,  color=color,  thickness=1)

                if configs.img_channel == 1:
                    frame = np.repeat(frame, 3).reshape(res_height, res_width*2, 3)

                writer.writeFrame(frame)
            writer.close()

            img = np.maximum(img, 0)
            img = np.minimum(img, 1)
            img_name = os.path.join(res_path, str(batch_id) + '.png')
            cv2.imwrite(img_name, (img * 255).astype(np.uint8))

    img_mse   /= num_val
    img_psnr  /= num_val
    img_ssim  /= num_val
    #img_lpips /= num_val
    avg_mse   = np.mean(img_mse).item()
    avg_psnr  = np.mean(img_psnr).item()
    avg_ssim  = np.mean(img_ssim).item()
    #avg_lpips = np.mean(img_lpips).item()

    summary = {
        "mse_avg": avg_mse,
        "psnr_avg": avg_psnr,
        "ssim_avg": avg_ssim,
        #"lpips_avg": avg_lpips,
        "mse": img_mse.tolist(),
        "psnr": img_psnr.tolist(),
        "ssim": img_ssim.tolist(),
        #"lpips": img_lpips.tolist()
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
    print('----------------------------------------------------------------------------------------------------')
    print('|    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |    9    |    10   |')

    print('| -- *MSE  per frame: ' + str(avg_mse) + ' ------------------------------------------------------------')
    for i in range(output_length):
        print('| ' + str(img_mse[i])[:7] + ' ', end='')
    print('|')

    print('| --  *PSNR  per frame: ' + str(avg_psnr) + ' ---------------------------------------------------------')
    for i in range(output_length):
        print('| ' + str(img_psnr[i])[:7] + ' ', end='')
    print('|')

    print('| -- *SSIM per frame: ' + str(avg_ssim) + ' ----------------------------------------------------------')
    for i in range(output_length):
        print('| ' + str(img_ssim[i])[:7] + ' ', end='')
    print('|')

    """
    print('| -- *LPIPS per frame: ' + str(avg_lpips) + ' ----------------------------------------------------------')
    for i in range(output_length):
        print('| ' + str(img_lpips[i])[:7] + ' ', end='')
    print('|')
    print('----------------------------------------------------------------------------------------------------')
    print('| -- *LPIPS per frame: ' + str(avg_lpips) + ' ----------------------------------------------------------')
    """
