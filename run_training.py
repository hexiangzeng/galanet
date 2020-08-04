# -*- coding: utf-8 -*-
"""
# @Time    : May/23/2020
# @Author  : zhx
"""

import torch
import numpy as np
from multiprocessing import Pool
from time import time
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from paths import *
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.augmentations.utils import resize_segmentation
from utilites.split_trainval import make_split_file, load_split
from logger.logger_torch import Logger
from utilites.utils import print_to_log_file
from dataset import LungDataset, unpack_dataset, delete_npy, LungDatasetOffline
from utilites.softmax_helper import softmax_torch
from network_architecture import Unet, GALANet, GALANetv2, APLSNet, PDVNet, update_lr, \
    DiceLossCrossEntropy, dice_coff, DiceLossFocalLoss, SoftDiceLoss, \
    save_checkpoint, load_best_checkpoint, load_latest_checkpoint
from inference import predict_preprocessed_data_return_softmax, save_segmentation_nifti_from_softmax, \
    aggregate_scores, determine_postprocessing, visualize_prediction_and_reference
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

article = {"3DUnet":"(Published:31 May 2019 Journal of Digital Imaging):"
                    "Fully Automated Lung Lobe Segmentation in Volumetric Chest CT with 3D U-Net: Validation with Intra- and Extra-Datasets",
           "PDVNet":"(2018 CVPR):"
                     "Automatic segmentation of pulmonary lobes using a progressive dense V-network",
           "PPLS":"(2017 CVPR)"
                  "Pathological Pulmonary Lobe Segmentation from CT Images using Progressive Holistically Nested Neural Networks and Random Walker",
           "FRVNet":"(2018 International Joint Conference on Neural Network):"
                     "End-to-end supervised lung lobe segmentation",
           "APLNet":"(2019 ISBI: International Symposium on Biomedical Imaging)"
                     "Automatic Pulmonary Lobe Segmentation Using Deep Learning",
           "PLSNet":"(2019 arXiv: Image and Video Processing)"
                     "Efficient 3D Fully Convolutional Networks for Pulmonary Lobe Segmentation in CT Images",
           "RTSUNet":"(Published:15 May 2020 IEEE Transactions on Medical Imaging)"
                      "Relational Modeling for Robust and Efficient Pulmonary Lobe Segmentation in CT Scans"}


def val(net, val_loader, metrics, epoch=None, logger=None, tensorboard_summary=True, network_name="3DUNet", fold="1", is_PDVNet=False):
    net.eval()
    val_loss = []
    fore_ground_dice = []
    eval_tp = []
    eval_fp = []
    eval_fn = []
    with torch.no_grad():
        for ids, data, target in val_loader:
            data, target = data.float(), target.float()
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = net(data)
            # logger.image_summary("val_images/val_volume", data[0][0][data.shape[2]//2], epoch)  # 选最中间slice显示
            # logger.image_summary("val_images/val_segmentation", target[0][target.shape[1]//2], epoch)
            # out_softmax = torch.softmax(output, dim=1).detach().cpu()
            # logger.image_summary("val_images/val_prediction",
            #                      torch.argmax(out_softmax, dim=1)[0][output.shape[2] // 2], epoch)
            # logger.image_summary("val_images/val_pred_softmax",
            #                      out_softmax[0][0][output.shape[2] // 2], epoch)
            del data
            if is_PDVNet:
                target_shp = target.shape
                target = F.interpolate(target.unsqueeze(dim=1), size=(target_shp[1] // 2, target_shp[2] // 2,
                                                                      target_shp[3] // 2))[0]
            loss = metrics(output, target)

            val_loss.append(loss.detach().cpu().numpy())
            # conpute dice per classes
            fg_dice, tp_hard, fp_hard, fn_hard = dice_coff(output, target)
            fore_ground_dice.append(fg_dice)
            eval_tp.append(tp_hard)
            eval_fp.append(fp_hard)
            eval_fn.append(fn_hard)
            del target

    fore_ground_dice = np.mean(fore_ground_dice)
    eval_tp = np.sum(eval_tp, 0)
    eval_fp = np.sum(eval_fp, 0)
    eval_fn = np.sum(eval_fn, 0)
    global_dc_per_class = [np.round(i, 6) for i in [2 * i / (2 * i + j + k) for i, j, k in zip(eval_tp, eval_fp, eval_fn)]
                           if not np.isnan(i)]
    if tensorboard_summary and logger is not None:
        logger.scalar_summary('val_loss', np.mean(val_loss), epoch)
        logger.scalar_summary('valset dice/Average global foreground Dice', np.mean(global_dc_per_class), epoch)
        logger.scalar_summary('valset dice/fore_ground_dice', np.mean(fore_ground_dice), epoch)
        logger.scalar_summary('valset dice/tp', np.mean(eval_tp), epoch)
        logger.scalar_summary('valset dice/fp', np.mean(eval_fp), epoch)
        logger.scalar_summary('valset dice/fn', np.mean(eval_fn), epoch)
    print_to_log_file('Val set: Average loss: {:.6f}, Average global foreground Dice: '
                      '{}\n'.format(np.mean(val_loss), global_dc_per_class), network=network_name, fold=fold)

    return global_dc_per_class


def train(net, train_loader, optimizer, metrics, epoch, logger, is_PDVNet=False):
    net.train()
    training_loss = []
    for case_id, data, target in train_loader:
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        data, target = data.float(), target.float()

        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        # print("classes are contained in target variables:{}".format(torch.unique(target).cpu().numpy()))
        optimizer.zero_grad()

        output = net(data)
        # logger.image_summary("training_images/training_volume", data[0][0][data.shape[2]//2], epoch)
        # logger.image_summary("training_images/training_segmentation", target[0][target.shape[1]//2], epoch)
        # out_softmax = torch.softmax(output[2] if is_PDVNet else output, dim=1).detach().cpu()  # B,C,D,H,W
        # # print(output.shape, out_softmax.shape)
        # logger.image_summary("training_images/training_prediction",
        #                      torch.argmax(out_softmax, dim=1)[0][out_softmax.shape[2] // 2], epoch)
        # logger.image_summary("training_images/training_pred_softmax",
        #                      out_softmax[0][0][out_softmax.shape[2] // 2], epoch)
        del data
        if is_PDVNet:
            target_shp = target.shape  # B, D, H, W
            target = F.interpolate(target.unsqueeze(dim=1), size=(target_shp[1]//2, target_shp[2]//2,
                                                                  target_shp[3]//2))[0]
            m0 = metrics(output[0], target)
            m1 = metrics(output[1], target)
            m2 = metrics(output[2], target)
            loss = (m0 + m1 + m2) / 3
        else:
            loss = metrics(output, target)

        loss.backward()
        _ = clip_grad_norm_(net.parameters(), 12)
        optimizer.step()

        training_loss.append(loss.detach().cpu().numpy())
        del target
    logger.scalar_summary('training_loss', np.mean(training_loss), epoch)


def predict_validation(net, val_loader, patch_size, base_output_folder, default_num_threads=8, num_classes=6,
                       do_mirroring: bool = True, use_train_mode: bool = False, tiled: bool = True, step: int = 2,
                       save_softmax: bool = False, use_gaussian: bool = True, overwrite_pred: bool = False,
                       validation_folder_name: str = 'validation_prediction', debug: bool = False, all_in_gpu: bool = False,
                       force_separate_z: bool = None, interpolation_order: int = 3, interpolation_order_z: int = 0,
                       vis_slices: bool = True, ):
    assert os.path.basename(base_output_folder).startswith("fold"), "base_output_folder must start with fold ."
    fold = os.path.basename(base_output_folder)[-1]
    network_name = os.path.basename(os.path.dirname(base_output_folder))
    raw_output_folder = join(base_output_folder, validation_folder_name)
    maybe_mkdir_p(raw_output_folder)
    my_input_args = {'do_mirroring': do_mirroring,
                     'use_train_mode': use_train_mode,
                     'tiled': tiled,
                     'step': step,
                     'save_softmax': save_softmax,
                     'use_gaussian': use_gaussian,
                     'overwrite_pred': overwrite_pred,
                     'validation_folder_name': validation_folder_name,
                     'debug': debug,
                     'all_in_gpu': all_in_gpu,
                     'force_separate_z': force_separate_z,
                     'interpolation_order': interpolation_order,
                     'interpolation_order_z': interpolation_order_z,
                     }
    save_json(my_input_args, join(raw_output_folder, "validation_args.json"))
    mirror_axes = (0, 1, 2)
    pred_gt_tuples = []

    export_pool = Pool(default_num_threads)
    results = []
    dataset_info = val_loader.dataset.dataset_info
    gt_niftis_folder = join(raw_cropped_data_dir, "gt_segmentations")
    for k in val_loader.dataset.cases_lists:
        properties = dataset_info[k]['properties']
        fname = properties['list_of_data_files'].split("/")[-1].split(".nii.gz")[0]
        if overwrite_pred or (not isfile(join(raw_output_folder, fname + ".nii.gz"))) or \
                (save_softmax and not isfile(join(raw_output_folder, fname + ".npz"))):
            data = np.load(dataset_info[k]['data_file'])['data']
            print(k, data.shape)
            softmax_pred = predict_preprocessed_data_return_softmax(net, data[:-1], do_mirroring, num_repeats=1,
                                                                    use_train_mode=use_train_mode, batch_size=1,
                                                                    mirror_axes=mirror_axes, tiled=tiled,
                                                                    tile_in_z=True, step=step, patch_size=patch_size,
                                                                    use_gaussian=use_gaussian, all_in_gpu=all_in_gpu)
            if save_softmax:
                softmax_fname = join(raw_output_folder, fname + ".npz")
            else:
                softmax_fname = None
            if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                np.save(join(raw_output_folder, fname + ".npy"), softmax_pred)
                softmax_pred = join(raw_output_folder, fname + ".npy")
            case_export_args = (softmax_pred, join(raw_output_folder, fname+".nii.gz"), properties, interpolation_order,
                                None, None, None, softmax_fname, None, force_separate_z, interpolation_order_z)
            results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax, (case_export_args,)))
        pred_gt_tuples.append([join(raw_output_folder, fname + ".nii.gz"), join(gt_niftis_folder, fname + ".nii.gz")])

    _ = [i.get() for i in results]
    print_to_log_file("finished prediction", network=network_name, fold=fold)

    # evaluate raw predictions
    print_to_log_file("evaluation of raw predictions", network=network_name, fold=fold)
    _ = aggregate_scores(pred_gt_tuples, labels=list(range(num_classes)),
                         json_output_file=join(raw_output_folder, "summary.json"), num_threads=default_num_threads)
    print_to_log_file("determining postprocessing...", network=network_name, fold=fold)
    final_validation_folder_name = validation_folder_name+"_postprocessed"
    determine_postprocessing(base_output_folder, gt_niftis_folder, validation_folder_name,
                             final_subf_name=final_validation_folder_name, debug=debug,
                             advanced_postprocessing=False)

    # visualize validation with gif and slices format
    # reference folder from raw_splited_dir because volume and segmentation are in it.
    visualize_prediction_and_reference(base_output_folder, raw_splited_dir, final_validation_folder_name,
                                       vis_slices=vis_slices)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hyper-parameters management')
    # network
    parser.add_argument('network', default='3DUNet', type=str,
                        help='use which network.'
                             'must be one of following: 3DUNet, PDVNet, FRVNet, APLSNet, PLSNet, RTSUNet, GALANet')
    parser.add_argument('--metrics', type=str, default='dcce',
                        help='use what metrics ce(cross entropy), dc(dice loss), fl(focal loss)')  # dcce or dcfl
    # Hard-ware configurations
    # parser.add_argument('--cpu', action='store_true', default=False, help='use cpu only.')
    # parser.add_argument('--seed', type=int, default=2020, help='random seed for running.')

    # data I/O and dataset configurations
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='batch size of trainset')
    parser.add_argument('-val', "--validation_only", default=False, action="store_true",
                        help="use this if you want to only run the validation")
    parser.add_argument('--valbest', default=False, action='store_true', help='validate best ')
    parser.add_argument('-ps', '--patch_size', default="128, 192, 192", type=str, nargs="+",
                        help='patch size of train samples after resize')
    parser.add_argument('-fd', '--fold', type=int, default=1,
                        help='which folder(1-10) to train, all is to train 5 folder.')
    parser.add_argument('--offline_data', type=bool, default=False, help='whether to train using data offline.')
    parser.add_argument('-c', '--continue_training', action='store_true', default=False, help='continue to train')
    parser.add_argument('-uz', '--use_zip_data', action='store_true', default=False,
                        help='upzip the npz file in preprocessed folder. this can accelerate training.')
    parser.add_argument('--no_debug', action='store_true', default=False, help='continue to train')

    # train configurations
    parser.add_argument('--max_epochs', type=int, default=500, help='number od epochs to train (default: 500)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate (default: 0.01)')

    args = parser.parse_args()
    network = args.network
    assert network in ["3DUNet", "PDVNet", "FRVNet", "APLSNet", "PLSNet", "RTSUNet", "GALANet"], \
        "network must be one of following: 3DUnet, PDVNet, FRVNet, APLSNet, PLSNet, RTSUNet, GALANet"
    loss_func = args.metrics
    # assert loss_func=='dcce' or loss_func=='dcfl', "metrics must dcce or dcfl"
    batch_size = args.batch_size
    offline_data = args.offline_data
    fold = args.fold
    continue_training = args.continue_training
    use_zip = args.use_zip_data
    max_epochs = args.max_epochs
    lr = args.learning_rate
    validation_only = args.validation_only
    valbest = args.valbest
    if not use_zip:
        print("unpacking dataset")
        unpack_dataset(join(preprocessed_output_dir, preprocessed_data_identifer))
        print("Done.")
    if not os.path.isfile(join(preprocessed_output_dir, "trainval_splits.pkl")):
        make_split_file()

    assert fold <= 10, "fold must less than 10"
    output_folder = join(network_output_dir_base, network, "fold_%s" % str(args.fold))
    maybe_mkdir_p(output_folder)

    in_channels = 1
    num_classes = 5
    patch_size = [128, 460, 460]  # GALANet  # [128, 368, 472]  # PDVNet
    features_num_lists = [16, 32, 64, 128]  # GALANet
    # net_class = recursive_find_class(join(os.path.dirname(os.path.abspath(__file__)), "network_archtecture"),
    #                                  network, "galaNet.network_archtecture") # developing
    net = None
    if network == "3DUNet":
        if offline_data:
            net = Unet(in_channels, num_classes+1, patch_size=[128, 160, 160], features_num_lists=[24, 48, 96, 192, 384]).cuda()
        else:
            net = Unet(in_channels, num_classes + 1, patch_size=patch_size, features_num_lists=features_num_lists).cuda()
    elif network == "GALANet":
        if offline_data:
            net = GALANet(in_channels, num_classes+1, patch_size=[128, 192, 192], features_num_lists=[16, 32, 64, 128]).cuda()
        else:
            net = GALANet(in_channels, num_classes + 1, patch_size=patch_size, features_num_lists=features_num_lists).cuda()
    elif network == "APLSNet":
        if offline_data:
            net = APLSNet(in_channels, num_classes+1, patch_size=[128, 192, 192], features_num_lists=[32, 40, 48, 56]).cuda()
        else:
            net = APLSNet(in_channels, num_classes + 1, patch_size=patch_size, features_num_lists=features_num_lists).cuda()
    elif network == "PDVNet":
        if offline_data:
            net = PDVNet(in_channels, num_classes+1, patch_size=[128, 192, 192], base_features=24).cuda()
        else:
            net = PDVNet(in_channels, num_classes + 1, patch_size=patch_size, base_features=24).cuda()

    train_cases, val_cases = load_split(fold)
    if not offline_data:
        train_dataset = LungDataset(preprocessed_output_dir, train_cases, net.patch_size, True)
        val_dataset = LungDataset(preprocessed_output_dir, val_cases, net.patch_size, False)
    else:
        train_dataset = LungDatasetOffline(preprocessed_output_dir, fold, net.patch_size, True)
        val_dataset = LungDatasetOffline(preprocessed_output_dir, fold, net.patch_size, False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)

    optimizer = optim.SGD(net.parameters(), lr=lr)
    if loss_func == 'dcce':
        metrics = DiceLossCrossEntropy()
    elif loss_func == 'dcfl':
        metrics = DiceLossFocalLoss()
    elif loss_func == 'dc':
        metrics = SoftDiceLoss(apply_softmax=softmax_torch, square=True)
    else: raise ValueError("metrics argument error.")
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_to_log_file(train_cases, network=network, fold=fold)
    print_to_log_file(val_cases, network=network, fold=fold)
    print_to_log_file(net, network=network, fold=fold)
    print_to_log_file('Total number of parameters: %d' % num_params, network=network, fold=fold)

    log_dir = join(network_output_dir_base, "logger", network, "fold_"+str(fold))
    # if isdir(log_dir) and len(subfiles(log_dir)) > 0:
    #     shutil.rmtree(log_dir)
    maybe_mkdir_p(log_dir)
    logger = Logger(log_dir)

    if not args.no_debug and not validation_only:
        debug = {'network': network,
                 'batch_size': batch_size,
                 'patch_size': net.patch_size,
                 'features_num_lists': features_num_lists,
                 'train_cases': str(train_cases),
                 'val_cases': str(val_cases),
                 'fold': fold,
                 'continue_training': continue_training,
                 'use_zip': use_zip,
                 'max_epochs': max_epochs,
                 'lr': lr,
                 'validation_only': validation_only,
                 'valbest': valbest,
                 'optimizer': optimizer.state_dict(),
                 'network_archtecture': str(net),
                 'log_dir': log_dir,
                 'logger': str(logger),
                 'offline_data': offline_data
        }
        save_json(debug, join(output_folder, "debug.json"))

    torch.cuda.empty_cache()
    global_dc_per_class = [0]*num_classes
    if not validation_only:
        if not continue_training:
            latest_epoch = 0
        else:
            net, optimizer, latest_epoch = load_latest_checkpoint(output_folder, net, optimizer)
            assert latest_epoch <= max_epochs, "Have trained to final epochs{}.".format(max_epochs)
        for epoch in range(latest_epoch+1, max_epochs + 1):
            print_to_log_file("Epoch: %s"%epoch, network=network, fold=fold)
            epoch_start_time = time()
            update_lr(optimizer, lr, epoch, max_epochs)
            print_to_log_file("Learning rate:%s"%optimizer.param_groups[0]["lr"], network=network, fold=fold)
            train(net, train_loader, optimizer, metrics, epoch, logger, is_PDVNet=network=="PDVNet")
            new_global_dc_per_class = val(net, val_loader, metrics, epoch, logger,
                                          network_name=network, fold=fold, is_PDVNet=network=="PDVNet")
            epoch_end_time = time()
            print_to_log_file("This epoch took %f s.\n" % (epoch_end_time - epoch_start_time), network=network, fold=fold)
            if np.mean(global_dc_per_class) < np.mean(new_global_dc_per_class):
                global_dc_per_class = [i for i in new_global_dc_per_class]
                save_checkpoint(epoch, net, optimizer, join(output_folder, "model_best.model"))
            if epoch % 2 == 0:
                save_checkpoint(epoch, net, optimizer, join(output_folder, "model_latest.model"))
                # torch.save(net, join(net_save_dir, "model_latest.pkl"))  # Save net with parameters
        save_checkpoint(max_epochs, net, optimizer, join(output_folder, "model_final_checkpoint.model"))

    device = torch.device("cuda")
    if valbest:
        net, optimizer, epoch = load_best_checkpoint(output_folder, net, optimizer, train=False)
    else:
        net, optimizer, epoch = load_latest_checkpoint(output_folder, net, optimizer, train=False)
    net = net.to(device)
    global_dc_per_class = []
    n = 1
    for i in range(n):
        dc_per_val = val(net, val_loader, metrics, epoch, tensorboard_summary=False,
                         network_name=network, fold=fold, is_PDVNet=network=="PDVNet")
        global_dc_per_class.append(dc_per_val)
    global_dc_per_class = np.stack(global_dc_per_class, axis=0)
    global_dc_per_class = np.mean(global_dc_per_class, axis=0).round(6)
    print_to_log_file("The fold {}: {} times\n"
                      "\tAverage global foreground Dice:{}\n"
                      "\tValidation accuracy:{}.\n".format(fold, n, list(global_dc_per_class), np.mean(global_dc_per_class)),
                      network=network, fold=fold)

    print("Begin prediction validation and visualize prediction and reference.")
    # patch_size prediction but GALANet predict full resolution
    predict_validation(net, val_loader, patch_size, output_folder)
    # delete_npy(join(preprocessed_output_dir, preprocessed_data_identifer))


if __name__ == '__main__':
    main()

    # delete_npy(join(preprocessed_output_dir, preprocessed_data_identifer))
