import argparse
import csv
import os
import os.path as osp
import pickle

from mmcv.fileio import list_from_file
import SimpleITK as sitk
import cv2
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from tqdm import tqdm

root_path = "/home/public_data/zhangweiyi/meta_data"

def isInsideBbox(coord, bbox):
    # check if coord is inside bbox
    # coord is a tuple: (row, col)
    # bbox is a tuple: (min_row, min_col, max_row, max_col)
    assert len(coord) == 2
    assert len(bbox) == 4
    # those coord on the border is not considered inside the bbox
    if bbox[0] + 1 < coord[0] < bbox[2] - 1 and bbox[1] + 1 < coord[1] < bbox[3] - 1:
        return True
    else:
        return False


def processing_overlap_bbox(ann_masks):
    for idx, ann_mask in enumerate(ann_masks):
        ann_regions = regionprops(label(ann_mask))
        if len(ann_regions) > 0:
            contours, _ = cv2.findContours(ann_mask, mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
            for _ in ann_regions:
                if _.bbox_area > _.area:
                    coords_inside_bbox = []
                    for vertex in contours[0]:
                        if isInsideBbox((vertex[0, 1], vertex[0, 0]), _.bbox):
                            coords_inside_bbox.append(vertex)
                    if len(coords_inside_bbox) > 0:
                        coords_inside_bbox = np.row_stack(coords_inside_bbox)
                        ann_masks[idx] = cv2.fillPoly(
                            ann_mask, [coords_inside_bbox], 0)

    return ann_masks


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="csv file containing the DICOM dir and label file")
    parser.add_argument("--resultdir", type=str, required=True,
                        help="result dir")
    parser.add_argument("--result-file", type=str, required=True,
                        help="output path of result annotation pickle file")
    parser.add_argument("--low", type=float, default=0.05,
                        help="low percentile of windowing intensity, default: 0.05")
    parser.add_argument("--high", type=float, default=99.5,
                        help="high percentile of windowing intensity, default: 99.5")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="number of classes, default: 3")
    parser.add_argument("--overlapbbox", type=str, default="overlapbbox.list",
                        help="path to save overlap bbox list, default: overlapbbox.list")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.resultdir, exist_ok=True)
    anns = []
    all_labels = []
    slice_with_overlap_bbox = []
    with open(args.csv, "r") as f:
        reader = csv.reader(f)
        for subject in tqdm(reader, total=len(list_from_file(args.csv))):
            subject = subject[0]
            print(subject)
            ann_file = root_path+"/annotations/"+subject+"-label.nrrd"
            img_file_dir = root_path+"/imgs/"+subject
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(
                series_reader.GetGDCMSeriesFileNames(img_file_dir, recursive=True))
            img = series_reader.Execute()
            # windowing, by default we use 0.05~99.5 percentile
            low = np.percentile(sitk.GetArrayViewFromImage(img), args.low)
            high = np.percentile(sitk.GetArrayViewFromImage(img), args.high)
            img = sitk.IntensityWindowing(img, low, high)
            ann = sitk.ReadImage(ann_file)
            # convert to numpy array
            img = sitk.GetArrayFromImage(img).astype(np.uint8)
            ann = sitk.GetArrayFromImage(ann).astype(np.uint8)
            if img.shape != ann.shape:
                raise RuntimeError(
                    "unexpected shape error in {}".format(ann_file))
            for idx in range(ann.shape[0]):
                ann_regions = regionprops(label(ann[idx]))
                for _ in ann_regions:
                    if _.bbox_area > _.area:
                        filename = "{}-slice{}.png".format(
                            osp.basename(img_file_dir), idx)
                        slice_with_overlap_bbox.append(filename)
            ann = processing_overlap_bbox(ann)
            for idx, img_slice in enumerate(img):
                ann_slice = ann[idx]
                ann_dict = dict()
                filename = "{}-slice{}.png".format(
                    osp.basename(img_file_dir), idx)
                ann_dict['filename'] = filename
                ann_dict['height'] = img_slice.shape[0]
                ann_dict['width'] = img_slice.shape[1]
                Image.fromarray(img_slice).convert(mode='RGB').save(
                    osp.join(args.resultdir, filename))
                ann_regions = regionprops(label(ann_slice))
                # slices with object
                if len(ann_regions) > 0:
                    bboxes = []
                    labels = []
                    for _ in ann_regions:
                        # skimage bbox format [row_min, col_max, row_max, col_max]
                        # aka. [y_min, x_min, y_max, x_max]
                        # while mmdet middle format use [x1, y1, x2, y2]
                        # aka. [x_min, y_min, x_max, y_max]
                        coord = (_.bbox[1], _.bbox[0], _.bbox[3], _.bbox[2])
                        bboxes.append(coord)
                        v = ann_slice[int(_.coords[0, 0]),
                                      int(_.coords[0, 1])] - 1
                        if v not in range(args.num_classes):
                            raise ValueError(
                                "unexpected label value {}".format(v))
                        labels.append(v)
                    bboxes = np.array(bboxes, dtype=np.float32)
                    labels = np.array(labels, dtype=np.int64)
                # slices without object, set bboxes to (0, 4) array
                # and set labels to (0, ) array
                else:
                    bboxes = np.empty((0, 4), dtype=np.float32)
                    labels = np.empty((0,), dtype=np.int64)
                ann_dict['ann'] = {'bboxes': bboxes, 'labels': labels}
                anns.append(ann_dict)
    with open(args.result_file, "wb") as f:
        pickle.dump(anns, f)
    with open(args.overlapbbox, "w") as f:
        for line in slice_with_overlap_bbox:
            f.write(line + '\n')
