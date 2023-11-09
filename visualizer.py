import argparse
import os
from tqdm import tqdm

import numpy as np
import open3d as o3d
import torch

from vis import BoundingBox3D, LabelLUT, Visualizer


def get_pointcloud_fromfile(file_path, pts_feature_num):
    try:
        if file_path.endswith(".bin"):
            pts = np.fromfile(file_path, dtype=np.float32).reshape(-1, pts_feature_num)
        elif file_path.endswith(".pcd"):
            pcd = o3d.t.io.read_point_cloud(file_path)
            pts = pcd.point["positions"].numpy()
            for filed_name in ["intensity", "index", "timestamp"]:
                if filed_name in list(pcd.point):
                    pts = np.append(pts, pcd.point[filed_name].numpy(), axis=1)
        else:
            raise Exception("Unknown pointcloud file format.")
    except ValueError:
        print("Please check the data or the num of point cloud features.")
        return np.zeros((2, pts_feature_num), dtype=np.float32)
    else:
        return pts


def convert_bboxes(self, bboxes, labels=None, scores=None):
    num_samples = len(bboxes)
    ml3d_bboxes = []
    for i in range(num_samples):
        x, y, z, w, l, h, rz = bboxes[i][0:7]
        center = (x, y, z)
        direction_vecs = self._cal_front_left_up_vec(rz)
        size = (w, h, l)
        if labels is not None:
            if isinstance(labels[i], torch.Tensor):
                label_class = int(labels[i].item())
            else:
                label_class = int(labels[i])
        else:
            label_class = 0
        if scores is not None:
            if isinstance(scores[i], torch.Tensor):
                confidence = scores[i].item()
            else:
                confidence = scores[i]
        else:
            confidence = -1.0
        box = BoundingBox3D(center, *direction_vecs, size, rz, label_class, confidence)
        ml3d_bboxes.append(box)
    return ml3d_bboxes


def convert_data(*args, name, bboxes=None, labels=None, scores=None, images=None, param=None):
    # points = np.vstack(args)
    points_path = args[0]
    data = {"name": name, "point": points_path}
    if bboxes is not None and len(bboxes) > 0:
        if isinstance(bboxes[0], BoundingBox3D):
            data["bounding_boxes"] = bboxes
        else:
            data["bounding_boxes"] = convert_bboxes(bboxes, labels, scores)
    if labels is not None:
        data["labels"] = labels
    if images is not None and len(images) > 0:
        data["cams"] = images
        if param is not None:
            for cam in data["cams"]:
                data["cams"][cam].update(param[cam])
    return [data]


def get_data(data_root, pts_feature_num):
    pts_dir = f"{data_root}/bin"
    pts_files = sorted(os.listdir(pts_dir))
    data_list = []
    for pts_file in tqdm(pts_files):
        sample = pts_file[:-4]
        pts_path = f"{pts_dir}/{pts_file}"
        # points = get_pointcloud_fromfile(pts_path, pts_feature_num)
        points = pts_path
        data = convert_data(points, name=sample)
        data_list.extend(data)
    return data_list


def multiple_vis(data, pts_feature_num):
    vis = Visualizer()
    lut = LabelLUT()
    vis.visualize(data, lut, pts_feature_num=pts_feature_num, bounding_boxes=None, width=1500, height=900)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", help="Data root path.")
    parser.add_argument("--save_dir", default="data", help="Save path.")
    parser.add_argument("--task", default="od", choices=["od", "seg"], help="Task type, include od, seg.")
    parser.add_argument("--pts_feat_num", default=6, type=int, help="The number of point features.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data = get_data(args.data_root, args.pts_feat_num)
    multiple_vis(data, args.pts_feat_num)
