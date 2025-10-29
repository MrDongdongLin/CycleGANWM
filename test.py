"""
Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name winter2summer_cyclegan --model cyclegan_watermark

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
"""
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import gen_watermark_bits, get_acc
from torchvision.utils import save_image
from util.visualizer import Visualizer
from models.networks import StegaStampDecoder

import utils_img2 as utils_img
import numpy as np
import pandas as pd
import re
import torchvision.transforms as transforms
from PIL import Image

def tensor_to_image(tensor):
    image = tensor.permute(1, 2, 0).cpu().detach().numpy()
    image = (image + 1 ) /2 * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)

def tensor_to_image2(tensor):
    # mean = ([0.5] * 3)
    # std = ([0.5] * 3)
    # unorm = transforms.Normalize(
    #     mean=[-m / s for m, s in zip(mean, std)],
    #     std=[1 / s for s in std]
    # )
    output = tensor.cpu().detach().float().numpy()
    output = (np.transpose(output, (1, 2, 0)) + 1) / 2.0 * 255.0
    output = output.astype(np.uint8)

    return output

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.display_port = 8097
    opt.display_ncols = 4
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    if 'whit' in opt.decoder_path:
        Decoder = torch.jit.load(opt.decoder_path).to(model.device)
        Decoder.eval()
    else:
        Decoder = StegaStampDecoder(opt.crop_size, opt.input_nc, opt.watermark_size)
        Decoder = Decoder.cuda()
        Decoder.load_state_dict(torch.load(opt.decoder_path, map_location=model.device))
        Decoder.eval()
    
    # Freeze LDM and hidden decoder
    for param in Decoder.parameters():
        param.requires_grad = False

    acc_gan_A, acc_gan_B, acc_idt_A, acc_idt_B, acc_rec_A, acc_rec_B = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # save_dir = os.path.join(opt.test_outputs, opt.name)
    save_path = os.path.join(opt.test_outputs, opt.name, "robust")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    attacks = {
        # 'crop': {f'crop_{i:03}': lambda x, n=i: utils_img.center_crop(x, n) for i in np.arange(0.1, 0.5, 0.1)},
        # 'resize': {f'resize_{i:03}': lambda x, n=i: utils_img.resize(x, n) for i in np.arange(0.1, 0.5, 0.1)},
        # 'rotate': {f'rotate_{i:03}': lambda x, n=i: utils_img.rotate(x, n) for i in range(0, 361, 10)},
        'bright': {f'bright_{i:.2f}': lambda x, n=i: utils_img.adjust_brightness(x, n) for i in np.arange(0.0+1.0, 2.1+1.0, 0.2)},
        'contrast': {f'contrast_{i:.2f}': lambda x, n=i: utils_img.adjust_contrast(x, n) for i in np.arange(0.0+1.0, 2.1+1.0, 0.2)},
        'saturate': {f'saturate_{i:.2f}': lambda x, n=i: utils_img.adjust_saturation(x, n) for i in np.arange(0.0+1.0, 2.1+1.0, 0.2)},
        # 'sharpness': {f'sharpness_{i:.2f}': lambda x, n=i: utils_img.adjust_sharpness(x, n) for i in np.arange(0.0, 2.1, 0.2)},
        # 'hue': {f'hue_{i:.2f}': lambda x, n=i: utils_img.adjust_hue(x, n) for i in np.arange(-0.5, 0.51, 0.1)},
        'jpeg': {f'jpeg_{i:03}': lambda x, n=i: utils_img.jpeg_compress(x, n) for i in range(10, 101, 10)},
        'noise': {f'noise_{i:.2f}': lambda x, n=i: utils_img.gaussian_noise(x, n) for i in np.arange(0.0, 0.21, 0.02)},
        'blur': {f'blur_{i:03}': lambda x, n=i: utils_img.adjust_gaussian_blur(x, n) for i in range(1, 20, 2)},
        # 'comb': lambda x: utils_img.jpeg_compress(utils_img.adjust_brightness(utils_img.center_crop(x, 0.5), 1.5), 80),
    }

    fake_A_log_stats = {"iteration": {ii:{} for ii in range(len(dataset))}}
    fake_B_log_stats = {"iteration": {ii:{} for ii in range(len(dataset))}}
    fake_EA_log_stats = {"iteration": {ii:{} for ii in range(len(dataset))}}
    fake_EB_log_stats = {"iteration": {ii:{} for ii in range(len(dataset))}}

    bitacc_fptwhit_log_stats = {"iteration": {ii:{} for ii in range(1000)}}

    if not os.path.exists(os.path.join(opt.test_outputs, 'aug_imgs')):
        os.makedirs(os.path.join(opt.test_outputs, 'aug_imgs'))

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    epoch_iter = 0
    for i, data in enumerate(dataset):
        epoch_iter += opt.batch_size
        counter = i
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        if opt.model == 'cyclegan_wb':
            real_A = visuals['real_A'].to(model.device)
            real_B = visuals['real_B'].to(model.device)
            fake_A = visuals['fake_A'].to(model.device)
            fake_B = visuals['fake_B'].to(model.device)
            real_wA = model.real_wA.to(model.device)
            real_wB = model.real_wB.to(model.device)

            if opt.attack:
                for out_name, out_attack in attacks.items():
                    for name, attack in out_attack.items():
                        imgs_augs = utils_img.unnormalize_img(attack(utils_img.normalize_img(fake_A)))
                        factor = int(re.search(r'\d+', name).group()) if 'jpeg' in name else 100

                        img_list = []
                        for jj in range(imgs_augs.shape[0]):
                            img_str = f'bs_{i:03}_iter_{jj:03}_test_w_{name}.jpeg' if 'jpeg' in name else f'bs_{i:03}_iter_{jj:03}_test_w_{name}.png'
                            image = tensor_to_image(imgs_augs[jj])
                            image = Image.fromarray(image, mode='RGB')
                            image.save(os.path.join(opt.test_outputs, 'aug_imgs', img_str), quality=factor)
                            image = Image.open(os.path.join(opt.test_outputs, 'aug_imgs', img_str)).convert('RGB')
                            image = preprocess(image)
                            img_list.append(image)
                        imgs_aug = torch.stack(img_list).to(model.device)
                        fpt_wm_whit = Decoder(imgs_aug)
                        # fpt_wm = decoder(utils_img.unnormalize_img(attack(utils_img.normalize_img(fpt_imgs))))

                        ori_msgs = torch.sign(real_wB) > 0
                        # decoded_msgs = torch.sign(fpt_wm) > 0  # b k -> b k
                        decoded_msgs_whit = torch.sign(fpt_wm_whit) > 0  # b k -> b k
                        # diff = (~torch.logical_xor(ori_msgs, decoded_msgs))  # b k -> b k
                        diff_whit = (~torch.logical_xor(ori_msgs, decoded_msgs_whit))  # b k -> b k
                        # bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # b k -> b
                        bit_accs_whit = torch.sum(diff_whit, dim=-1) / diff_whit.shape[-1]  # b k -> b
                        for jj in range(bit_accs_whit.shape[0]):
                            img_num = i * opt.batch_size + jj
                            # log_stat = bitacc_fpt_log_stats["iteration"][img_num]
                            # log_stat[f'bit_acc_{name}'] = bit_accs[jj].item()

                            log_stat = bitacc_fptwhit_log_stats["iteration"][img_num]
                            log_stat[f'bit_acc_{name}'] = bit_accs_whit[jj].item()

            trainAB_path = os.path.join(opt.test_outputs, 'trainAB')
            trainBA_path = os.path.join(opt.test_outputs, 'trainBA')
            testAB_path = os.path.join(opt.test_outputs, 'testAB')
            testBA_path = os.path.join(opt.test_outputs, 'testBA')
            if not os.path.exists(trainAB_path):
                os.makedirs(trainAB_path)
            if not os.path.exists(trainBA_path):
                os.makedirs(trainBA_path)
            if not os.path.exists(testAB_path):
                os.makedirs(testAB_path)
            if not os.path.exists(testBA_path):
                os.makedirs(testBA_path)
            from torchvision import utils as vutils
            os.makedirs(os.path.join(testAB_path, "single"), exist_ok=True)
            if i < opt.num_test -200:
                direction = 'AtoB'
                vutils.save_image(fake_B, os.path.join(testAB_path, "single", f'ABid_{i}_images.jpg'), nrow=opt.batch_size)

                visualizer.save_paired_img(trainAB_path, real_A, fake_B, i, direction)
                direction = 'BtoA'
                visualizer.save_paired_img(trainBA_path, real_B, fake_A, i, direction)
            elif i>9000000: #i > 0 and i < opt.num_test:
                direction = 'AtoB'
                visualizer.save_paired_img(testAB_path, real_A, fake_B, i, direction)
                direction = 'BtoA'
                visualizer.save_paired_img(testBA_path, real_B, fake_A, i, direction)

            # bitacc_gan_A, bitacc_gan_B, bitacc_idt_A, bitacc_idt_B, bitacc_rec_A, bitacc_rec_B = visualizer.print_current_bitacc(
            #     model, 0, epoch_iter, opt.batch_size, opt.phase)
            bitacc_gan_A = get_acc(opt, fake_B, Decoder, real_wA)
            bitacc_gan_B = get_acc(opt, fake_A, Decoder, real_wB)

            with open(os.path.join(opt.test_outputs, opt.name, opt.phase + '_onepic_bitacc_log.txt'), 
                      "a") as log_file:
                log_file.write('%f %f\n' % (bitacc_gan_A, bitacc_gan_B))


            acc_gan_A += bitacc_gan_A
            acc_gan_B += bitacc_gan_B

        if opt.model == 'pix2pix':
            real_A = visuals['real_A'].to(model.device)
            fake_B = visuals['fake_B'].to(model.device)
            this_batch_size = min(opt.batch_size, real_A.size(0))
            real_wA, real_wB = gen_watermark_bits(opt, this_batch_size, model.device)

            testABpp_path = os.path.join(opt.test_outputs, 'testABpp')
            testBApp_path = os.path.join(opt.test_outputs, 'testBApp')

            from torchvision import utils as vutils
            if not os.path.exists(testABpp_path):
                os.makedirs(testABpp_path)
            if not os.path.exists(testBApp_path):
                os.makedirs(testBApp_path)
            os.makedirs(os.path.join(testABpp_path, "single"), exist_ok=True)
            if opt.test_direction == 'AtoB':
                vutils.save_image(fake_B, os.path.join(testABpp_path, "single", f'ABid_{i}_images.jpg'), nrow=opt.batch_size)
                # visualizer.save_paired_img(testABpp_path, real_A, fake_B, i, opt.test_direction)
                bitacc_gan_A = get_acc(opt, fake_B, Decoder, real_wA)
            else:
                visualizer.save_paired_img(testBApp_path, real_A, fake_B, i, opt.test_direction)
                bitacc_gan_B = get_acc(opt, fake_B, Decoder, real_wB)

    bitacc_fptwhit_log_stats = [{'img': each_img, **bitacc_fptwhit_log_stats["iteration"][each_img]} for each_img in range(opt.num_test)]
    print(f'>>> Saving log stats to {opt.test_outputs}...')
    df_fake_A = pd.DataFrame(bitacc_fptwhit_log_stats)
    df_fake_A.to_csv(os.path.join(opt.test_outputs, 'val_bitacc_fptwhit_log_stats_whit.csv'), index=False)
    print(df_fake_A)

    avg_gan_A = acc_gan_A / counter
    avg_gan_B = acc_gan_B / counter

    message = '(Test epoch: %d, iters: %d, bitacc_gan_A: %.3f, bitacc_gan_B: %.3f) ' % (
        0, epoch_iter, avg_gan_A, avg_gan_B)

    print(message)  # print the message
    with open(os.path.join(opt.test_outputs, opt.phase + '_bitacc_log.txt'),
              "a") as log_file:
        log_file.write('%s\n' % message)  # save the message

    webpage.save()  # save the HTML
