# -*- coding: utf-8 -*-
"""
# @Time    : May/18/2020
# @Author  : zhx
"""

import numpy as np
import SimpleITK as sitk
import imageio
from multiprocessing import Pool
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from batchgenerators.utilities.file_and_folder_operations import *

COLOR = {
    '1': (128, 174, 128), '2': (241, 214, 145),
    '3': (177, 122, 101), '4': (111, 184, 210), '5': (216, 101, 79)
}

DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3


def hu_to_grayscale(volume, hu_min=DEFAULT_HU_MIN, hu_max=DEFAULT_HU_MAX):
    # Clip at max and min values if specified
    # if hu_min is not None or hu_max is not None:
    #     volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


# Based on: https://github.com/neheller/kits19/blob/master/starter_code/visualize.py
def overlay_segmentation(vol, seg, alpha=DEFAULT_OVERLAY_ALPHA):
    # Scale volume to greyscale range
    # if isinstance(vol, sitk.Image):
    #     vol = sitk.GetArrayFromImage(vol)
    # if isinstance(seg, sitk.Image):
    #     seg = sitk.GetArrayFromImage(seg)
    # vol_greyscale = (255*(vol - np.min(vol))/np.ptp(vol)).astype(int)
    # vol_greyscale = np.stack([vol_greyscale, vol_greyscale, vol_greyscale], axis=-1)
    vol_greyscale = hu_to_grayscale(vol)
    # Convert volume to RGB
    # Initialize segmentation in RGB
    shp = seg.shape
    seg_rgb = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.int)
    # Set class to appropriate color
    classes = [i for i in np.unique(seg) if i != 0]
    assert len(classes) < 7, 'sorry, i set 5 foreground color only because of my lazy.'
    for c in classes:
        seg_rgb[np.equal(seg, c)] = COLOR[str(c)]
    # Get binary array for places where an ROI lives
    segbin = np.greater(seg, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    vol_overlayed = np.where(
        repeated_segbin,
        np.round(alpha*seg_rgb+(1-alpha)*vol_greyscale).astype(np.uint8),
        np.round(vol_greyscale).astype(np.uint8)
    )
    # Return final volume with segmentation overlay
    return vol_overlayed


def visualize_val_animation(case_id, vol_img, seg_img, pred_img, gif_path,
                            slices_out_folder, alpha=DEFAULT_OVERLAY_ALPHA):
    assert gif_path.endswith(".gif"), "gif_path must end with .gif"
    # Color volumes according to truth and pred segmentation
    vol = sitk.GetArrayFromImage(vol_img)
    seg = sitk.GetArrayFromImage(seg_img)
    pred = sitk.GetArrayFromImage(pred_img)
    vol_truth = overlay_segmentation(vol, seg, alpha)[::-1]
    vol_pred = overlay_segmentation(vol, pred, alpha)[::-1]

    # Create a figure and two axes objects from matplot
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Initialize the two subplots (axes) with an empty 512x512 image
    data = np.zeros(vol.shape[1:3])
    ax1.set_title("Ground Truth")
    ax2.set_title("Prediction")
    img1 = ax1.imshow(data)
    img2 = ax2.imshow(data)

    fnames = seg.shape[1]
    # Update function for both images to show the slice for the current frame
    def update(i):
        plt.suptitle("Case ID: " + str(case_id) + " - " + "Slice: " + str(i))
        img1.set_data(vol_truth[:, :, i])
        img2.set_data(vol_pred[:, :, i])
        return [img1, img2]

    # Compute the animation (gif)
    ani = animation.FuncAnimation(fig, update, frames=fnames, interval=5,
                                  repeat_delay=0, blit=False)

    # Save the animation (gif)
    ani.save(gif_path, writer='imagemagick', fps=30)
    # Close the matplot
    plt.close()

    if slices_out_folder is not None:
        maybe_mkdir_p(slices_out_folder)
        # save slice to slice case id path
        seg_path = join(slices_out_folder, "reference")
        maybe_mkdir_p(seg_path)
        pred_path = join(slices_out_folder, "prediction")
        maybe_mkdir_p(pred_path)

        for i in range(vol_truth.shape[0]):
            imageio.imsave(join(seg_path, "{:05d}.png".format(i)), vol_truth[i])
            imageio.imsave(join(pred_path, "{:05d}.png".format(i)), vol_pred[i])
        del vol, vol_truth, vol_pred, seg, pred


def visualize_single_case(args):
    case_id, prediction_folder, reference_folder, gif_out_folder, case_slices_folder, is_val, overwrite = args
    print("Processing {}".format(case_id))
    pred_path = join(prediction_folder, case_id + ".nii.gz")
    preffix_name = "train" if is_val else "test"
    img_path = join(reference_folder, preffix_name+"_images", case_id + ".nii.gz")
    seg_path = join(reference_folder, preffix_name+"_labels", case_id + ".nii.gz")

    if case_slices_folder is not None:
        case_slices_folder = join(case_slices_folder, case_id)

    gif_path = join(gif_out_folder, str(case_id) + ".gif")
    if not overwrite and isfile(gif_path):
        if case_slices_folder is None or isdir(case_slices_folder):
            print("Have processed {}.".format(case_id))
            return
    if case_slices_folder is not None: maybe_mkdir_p(case_slices_folder)
    try:
        vol_img = sitk.ReadImage(img_path)
        seg_img = sitk.ReadImage(seg_path)
        pred_img = sitk.ReadImage(pred_path)  # depth, height, width
        visualize_val_animation(case_id, vol_img, seg_img, pred_img, gif_path, case_slices_folder)
    except Exception as e:
        print(e)
        raise "### 错误 {}\t{}\n".format(case_id, pred_path)
    print("Have precessed {}.\n".format(case_id))


def visualize_prediction_and_reference(base_output_folder, reference_folder,
                                       final_prediction_folder_name="validation_prediction",
                                       visualize_gif_folder_name="visualize_gif",
                                       visualize_slices_folder_name="visualize_slices",
                                       is_val=True, vis_slices=False,
                                       default_num_processesor=8, overwrite_gif=False):
    assert os.path.basename(base_output_folder).startswith("fold"), "visualize pred and ref must in fold dir."
    final_pred_folder = join(base_output_folder, final_prediction_folder_name)
    case_identifer = [c[:-7] for c in subfiles(final_pred_folder, join=False, suffix=".nii.gz", sort=True)]
    # visualize output path
    gif_out_folder = join(base_output_folder, visualize_gif_folder_name)
    maybe_mkdir_p(gif_out_folder)
    slices_out_folder = None
    if vis_slices:
        slices_out_folder = join(base_output_folder, visualize_slices_folder_name)
        maybe_mkdir_p(slices_out_folder)
    p = Pool(default_num_processesor)
    args = (case_identifer, [final_pred_folder]*len(case_identifer), [reference_folder]*len(case_identifer),
            [gif_out_folder]*len(case_identifer), [slices_out_folder]*len(case_identifer), [is_val]*len(case_identifer),
            [overwrite_gif]*len(case_identifer))
    p.map(visualize_single_case, zip(*args))
    p.close()
    p.join()


# coded from wjcy
def hu2gray(volume, WL=-600, WW=900):
    low = WL - 0.5 * WW
    volume = (volume - low) / WW * 255.0
    volume[volume > 255] = 255
    volume[volume < 0] = 0
    volume = np.uint8(volume)
    return volume


if __name__ == '__main__':
    case_id = "ME07121214"
    vol = sitk.ReadImage("/media/zhx/My Passport/lung_lobe_seg/galaNet_raw_data/splited_data/"
                         "train_images/{}.nii.gz".format(case_id))
    seg = sitk.ReadImage("/media/zhx/My Passport/lung_lobe_seg/galaNet_raw_data/splited_data/"
                         "train_labels/{}.nii.gz".format(case_id))
    pred = sitk.ReadImage("/media/zhx/My Passport/lung_lobe_seg/galaNet_trained_models/GALANet/"
                          "fold_1/validation_prediction_postprocessed/{}.nii.gz".format(case_id))
    gif_path = "/media/zhx/My Passport/lung_lobe_seg/test_{}.gif".format(case_id)
    scans_path = "/media/zhx/My Passport/lung_lobe_seg/{}".format(case_id)
    visualize_val_animation("XXXXXtestXXXXX", vol, seg, pred, gif_path, scans_path)
