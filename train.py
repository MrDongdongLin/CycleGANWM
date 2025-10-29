"""
Example:
    Train a CycleGAN watermark model:
        python train.py --dataroot /pubdata/ldd/winter2summer --name winter2summer_cyclegan --model cyclegan_wb
    Train a CycleGAN surrogate model:
        python train.py --dataroot /pubdata/ldd/winter2summer --name winter2summer_cyclegan_surrogate --model cycle_gan
"""
import time
import os
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import gen_watermark_bits, get_acc
from models.networks import StegaStampDecoder
import pandas as pd

import torch.nn as nn
import numpy as np

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # for testing
    if opt.model == 'cyclegan_wb':
        # bs = opt.batch_size
        opt.phase = 'test'  # chosen from test set
        opt.max_dataset_size = 500
        # opt.batch_size = 16
        val_dataset = create_dataset(opt)
        opt.phase = 'train'
        # opt.batch_size = bs
    if opt.model == 'pix2pix':
        if opt.data_direction == 'AtoB':
            opt.phase = 'testAB'  # chosen from test set
            # opt.batch_size = 16
            val_dataset = create_dataset(opt)
            opt.phase = 'trainAB'
        else:
            opt.phase = 'testBA'  # chosen from test set
            # opt.batch_size = 16
            val_dataset = create_dataset(opt)
            opt.phase = 'trainBA'
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    train_df = pd.DataFrame(columns=[
        "epoch",
        "iteration",
        "bit_acc_avg_ganw_a2b",
        "bit_acc_avg_ganw_b2a",
        "bit_acc_avg_idtw_a2a",
        "bit_acc_avg_idtw_b2b",
        "bit_acc_avg_cycw_a2b2a",
        "bit_acc_avg_cycw_b2a2b",
    ])

    test_df = pd.DataFrame(columns=[
        "epoch",
        "iteration",
        "bit_acc_avg_ganw_a2b",
        "bit_acc_avg_ganw_b2a",
        "bit_acc_avg_idtw_a2a",
        "bit_acc_avg_idtw_b2b",
        "bit_acc_avg_cycw_a2b2a",
        "bit_acc_avg_cycw_b2a2b",
    ])

    save_df = pd.DataFrame(columns=[
        "epoch",
        "bit_acc_avg_ganw_a2b",
        "bit_acc_avg_ganw_b2a",
    ])

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # print bitwise accuracy for training and testing
            
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % 1 == 0:
            if opt.model == 'cyclegan_wb':
                bitacc_gan_A, bitacc_gan_B, bitacc_idt_A, bitacc_idt_B, bitacc_rec_A, bitacc_rec_B = \
                    visualizer.print_current_bitacc(model, epoch, epoch_iter, opt.batch_size, opt.phase)
                log_stats = {
                    "epoch": epoch,
                    "iteration": i,
                    "bit_acc_avg_ganw_a2b": bitacc_gan_A,
                    "bit_acc_avg_ganw_b2a": bitacc_gan_B,
                    "bit_acc_avg_idtw_a2a": bitacc_idt_A,
                    "bit_acc_avg_idtw_b2b": bitacc_idt_B,
                    "bit_acc_avg_cycw_a2b2a": bitacc_rec_A,
                    "bit_acc_avg_cycw_b2a2b": bitacc_rec_B,
                }
                train_df = train_df._append({**log_stats}, ignore_index=True)
                opt.phase = 'test'
                # bs = opt.batch_size
                # opt.batch_size = 10
                # acc_gan_A, acc_gan_B, acc_idt_A, acc_idt_B, acc_rec_A, acc_rec_B = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for j, val_data in enumerate(val_dataset):
                    # counter = j
                    # if j > opt.num_val:  # only apply our model to opt.num_test images.
                    #     break
                    model.set_input(val_data)  # unpack data from data loader
                    model.test()  # run inference
                    if j % opt.display_freq == 0:
                        bitacc_gan_A, bitacc_gan_B, bitacc_idt_A, bitacc_idt_B, bitacc_rec_A, bitacc_rec_B = \
                            visualizer.print_current_bitacc(model, epoch, epoch_iter, opt.batch_size, opt.phase)
                        log_stats = {
                            "epoch": epoch,
                            "iteration": i,
                            "bit_acc_avg_ganw_a2b": bitacc_gan_A,
                            "bit_acc_avg_ganw_b2a": bitacc_gan_B,
                            "bit_acc_avg_idtw_a2a": bitacc_idt_A,
                            "bit_acc_avg_idtw_b2b": bitacc_idt_B,
                            "bit_acc_avg_cycw_a2b2a": bitacc_rec_A,
                            "bit_acc_avg_cycw_b2a2b": bitacc_rec_B,
                        }
                        test_df = test_df._append({**log_stats}, ignore_index=True)
                opt.phase = 'train'
                # opt.batch_size = bs
                print(train_df)
                print(test_df)
            if opt.model == 'pix2pix':
                # python train.py --dataroot /pubdata/ldd/landscape/surrogate --gpu_ids 0 --dataset_mode aligned --phase trainAB --direction AtoB
                results = model.get_current_visuals()
                real_A = results['real_A']
                fake_B = results['fake_B']
                real_B = results['real_B']
                this_batch_size = min(opt.batch_size, real_A.size(0))
                real_wA, real_wB = gen_watermark_bits(opt, this_batch_size, model.device)

                Decoder = StegaStampDecoder(opt.crop_size, opt.input_nc, opt.watermark_size)
                Decoder = Decoder.cuda()
                Decoder.load_state_dict(torch.load(opt.decoder_path, map_location=model.device))
                Decoder.eval()

                bitacc_gan_wA = get_acc(opt, fake_B, Decoder, real_wA)
                bitacc_gan_wB = get_acc(opt, fake_B, Decoder, real_wB)

                message = '(Validate epoch: %d, iters: %d, bitacc_gan_wA: %.3f, bitacc_gan_wB: %.3f) ' % (
                    epoch, epoch_iter, bitacc_gan_wA, bitacc_gan_wB)
                print(message)
                with open(os.path.join(opt.checkpoints_dir, opt.name, opt.phase + '_bitacc_log.txt'),
                            "a") as log_file:
                    log_file.write('%s\n' % message)  # save the message

                if opt.data_direction == 'AtoB':
                    opt.phase = 'testAB'
                else:
                    opt.phase = 'testBA'
                acc_gan_wA, acc_gan_wB = 0.0, 0.0
                for j, val_data in enumerate(val_dataset):
                    counter = j
                    if j > opt.num_val:  # only apply our model to opt.num_test images.
                        break
                    model.set_input(val_data)  # unpack data from data loader
                    model.test()  # run inference

                    results = model.get_current_visuals()
                    real_A = results['real_A']
                    fake_B = results['fake_B']
                    real_B = results['real_B']
                    this_batch_size = min(opt.batch_size, real_A.size(0))

                    bitacc_gan_wA = get_acc(opt, fake_B, Decoder, real_wA)
                    bitacc_gan_wB = get_acc(opt, fake_B, Decoder, real_wB)

                    acc_gan_wA += bitacc_gan_wA
                    acc_gan_wB += bitacc_gan_wB

                avg_gan_wA = acc_gan_wA / counter
                avg_gan_wB = acc_gan_wB / counter

                message = '(Test epoch: %d, iters: %d, bitacc_gan_wA: %.3f, bitacc_gan_wB: %.3f) ' % (
                    epoch, epoch_iter, avg_gan_wA, avg_gan_wB)
                
                save_log = {
                    "epoch": epoch,
                    "bit_acc_avg_ganw_a2b": avg_gan_wA,
                    "bit_acc_avg_ganw_b2a": avg_gan_wB,
                }
                save_df = save_df._append({**save_log}, ignore_index=True)
                
                print(message)
                with open(os.path.join(opt.checkpoints_dir, opt.name, opt.data_name + opt.phase + '_bitacc_log.txt'),
                            "a") as log_file:
                    log_file.write('%s\n' % message)  # save the message
                
                from torchvision import utils as vutils
                if epoch % 20 == 0:
                    save_path = os.path.join(opt.checkpoints_dir, opt.name, opt.data_name + opt.phase)
                    os.makedirs(save_path, exist_ok=True)
                    merged_images = torch.cat((real_A, fake_B, fake_B), dim=0)
                    vutils.save_image(merged_images, os.path.join(save_path, f'epoch_{epoch}_images.jpg'), nrow=opt.batch_size)

                    # visualizer.save_paired_img(save_path, real_A, fake_B, i, opt.test_direction)
                    save_df.to_csv(os.path.join(opt.checkpoints_dir, opt.name, opt.data_name + opt.phase + f'_bitacc_log.csv'), index=False)


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

            train_df.to_csv(os.path.join(opt.checkpoints_dir, opt.name, f'train_log_stats_epoch_{epoch:03d}.csv'), index=False)
            test_df.to_csv(os.path.join(opt.checkpoints_dir, opt.name, f'test_log_stats_epoch_{epoch:03d}.csv'), index=False)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
