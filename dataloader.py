import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from pathlib import Path
import numpy as np
from collections import deque
import yaml
from utility import latlon_to_yaw, euler_from_quaternion, transform_2d_points, resizecrop_matrix, crop_matrix, cls2one_hot, colorize_depth
from config import GlobalConfig

config = GlobalConfig()

class KarrDataset(Dataset):
    def __init__(self, split='train'):
        self.config = config
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.data_rate = self.config.data_rate
        self.rp1_close = self.config.rp1_close
        self.root_dir = Path(self.config.root_dir)
        self.split = split

        self.rgb = []
        self.seg = []
        self.pcd = []
        self.lon = []
        self.lat = []
        self.local_x = []
        self.local_y = []
        self.rp1_lon = []
        self.rp1_lat = []
        self.rp2_lon = []
        self.rp2_lat = []
        self.bearing = []
        self.local_heading = []
        self.gnss_heading = []
        self.imu_heading = []
        self.velocity = []

        self.data = None

        # meta
        self.dir_meta = self.root_dir / split / "meta"

        # rgbd directory
        self.dir_rgb_front = self.root_dir / split / "camera" / "rgb"
        self.dir_rgb_seg = self.root_dir / split / "camera" / "seg" / "map"
        self.dir_depth_front = self.root_dir / split / "camera" / "depth" / "map"
        self.dir_histo = self.root_dir / split / "camera" / "histogram"
        self.dir_optflow = self.root_dir / split / "camera" / "optical_flow"

        self.files = os.listdir(self.dir_meta)
        self.files.sort()
        self.files = [os.path.splitext(filename)[0] for filename in self.files] # remove extension string
        self.len_files = len(self.files)

        with open(f"{self.root_dir}/{split}/routepoint_{split}.yml", "r") as f:
            rp_list = yaml.safe_load(f)
            rp_list['route_point']['latitude'].append(rp_list['last_point']['latitude'])
            rp_list['route_point']['longitude'].append(rp_list['last_point']['longitude'])

        # Initialize prev_lat/prev_lon from the frame just before the first "current" frame
        # so latlon_to_yaw has valid input even on the very first iteration.
        _first_current_file = self.files[self.seq_len - 2] if self.seq_len >= 2 else self.files[0]
        with open(f"{self.dir_meta}/{_first_current_file}.yml", "r") as _f:
            _meta_init = yaml.safe_load(_f)
        prev_lat = _meta_init["global_position_latlon"][0]
        prev_lon = _meta_init["global_position_latlon"][1]

        # sequences for past and current frames
        for i in range(0, self.len_files - (self.seq_len - 1) - (self.pred_len * self.data_rate)):
            filename = ""
            rgbs = []
            segs = []
            pcds = []
            local_xs = []
            local_ys = []
            local_headings = []

            # read files sequentially (past and current frames)
            for j in range(0, self.seq_len):
                filename = self.files[i+j]
                rgbs.append(f"{self.dir_rgb_front}/{filename}.png")
                segs.append(f"{self.dir_rgb_seg}/{filename}.png")
                pcds.append(f"{self.dir_depth_front}/{filename}.npy")
            self.rgb.append(rgbs)
            self.seg.append(segs)
            self.pcd.append(pcds)

            # get local loc, heading, vehicular controls, gps loc, and bearing at current frame (last of sequence)
            with open(f"{self.dir_meta}/{filename}.yml", "r") as f:
                meta_current = yaml.safe_load(f)

            local_xs.append(meta_current["local_position_xyz"][0])
            local_ys.append(meta_current["local_position_xyz"][1])
            local_quaternion = meta_current["local_orientation_xyzw"]
            local_headings.append(euler_from_quaternion(local_quaternion[3], local_quaternion[0], local_quaternion[1], local_quaternion[2], rad=True)[2])
            curr_lat = meta_current["global_position_latlon"][0]
            curr_lon = meta_current["global_position_latlon"][1]
            self.lat.append(curr_lat)
            self.lon.append(curr_lon)
            self.velocity.append(np.abs(meta_current["velocity"]))

            if np.abs(meta_current["velocity"]) > 0.5:
                bearing_latlon = latlon_to_yaw(
                    curr_lat, curr_lon, prev_lat, prev_lon,
                    offset=0.0
                    )
                self.bearing.append(bearing_latlon)
            else:
                _, _, bearing_witmotion = euler_from_quaternion(
                    w=meta_current['global_orientation_xyzw'][3],
                    x=meta_current['global_orientation_xyzw'][0],
                    y=meta_current['global_orientation_xyzw'][1],
                    z=meta_current['global_orientation_xyzw'][2],
                    rad=True
                    )
                bearing_witmotion = np.degrees(bearing_witmotion) - 90
                bearing_witmotion = np.radians(bearing_witmotion)
                self.bearing.append(bearing_witmotion)
            
            prev_lat, prev_lon = curr_lat, curr_lon

            # assign next route lat lon (rp1, rp2)
            about_to_finish = False

            for j in range(2):
                next_lat_rp = rp_list["route_point"]["latitude"][j]
                next_lon_rp = rp_list["route_point"]["longitude"][j]
                dLat_m = (next_lat_rp - meta_current["global_position_latlon"][0]) * 40008000 / 360
                dLon_m = (next_lon_rp - meta_current['global_position_latlon'][1]) * 40075000 * np.cos(np.radians(meta_current['global_position_latlon'][0])) / 360
                if j == 0 and np.sqrt(dLat_m**2 + dLon_m**2) <= self.rp1_close and not about_to_finish:
                    if len(rp_list['route_point']['latitude']) > 2:
                        rp_list['route_point']['latitude'].pop(0)
                        rp_list['route_point']['longitude'].pop(0)
                    else:
                        about_to_finish = True
                        rp_list['route_point']['latitude'][0] = rp_list['route_point']['latitude'][-1]
                        rp_list['route_point']['longitude'][0] = rp_list['route_point']['longitude'][-1]
                    next_lat_rp = rp_list['route_point']['latitude'][j]
                    next_lon_rp = rp_list['route_point']['longitude'][j]
                if j == 0:
                    self.rp1_lon.append(next_lon_rp)
                    self.rp1_lat.append(next_lat_rp)
                else:
                    self.rp2_lon.append(next_lon_rp)
                    self.rp2_lat.append(next_lat_rp)

            # read files sequentially (future frames for waypoints)
            for j in range(1, self.pred_len + 1):
                filename_future = self.files[(i + self.seq_len - 1) + (j * self.data_rate)]
                with open(f"{self.dir_meta}/{filename_future}.yml", "r") as read_meta_future:
                    meta_future = yaml.safe_load(read_meta_future)
                local_xs.append(meta_future["local_position_xyz"][0])
                local_ys.append(meta_future["local_position_xyz"][1])
                local_quaternion = meta_future["local_orientation_xyzw"]
                local_headings.append(euler_from_quaternion(local_quaternion[3], local_quaternion[0], local_quaternion[1], local_quaternion[2], rad=True)[2])

            self.local_x.append(local_xs)
            self.local_y.append(local_ys)
            self.local_heading.append(local_headings)
        
        print("Loaded Data Init")

    def __len__(self):
        return len(self.rgb)
    
    def __getitem__(self, index):
        data = dict()
        data["rgbs"] = []
        data['segs'] = []
        data['pcd_xs'] = []
        data['pcd_zs'] = []
        seq_rgbs = self.rgb[index]
        seq_segs = self.seg[index]
        seq_pcds = self.pcd[index]
        seq_local_xs = self.local_x[index]
        seq_local_ys = self.local_y[index]
        seq_local_headings = self.local_heading[index]

        for i in range(0, self.seq_len):
            rgb_img = cv2.imread(seq_rgbs[i])
            rgb_crop = crop_matrix(rgb_img, resize=self.config.scale, crop=self.config.crop_roi)
            rgb_transpose = rgb_crop.transpose(2, 0, 1)
            data["rgbs"].append(torch.from_numpy(np.array(rgb_transpose)))

            seg_img = cv2.imread(seq_segs[i])
            seg_crop = crop_matrix(seg_img, resize=self.config.scale, crop=self.config.crop_roi)
            seg_onehot = cls2one_hot(seg_crop, n_class=self.config.n_class)
            data["segs"].append(torch.from_numpy(np.array(seg_onehot)))

            pcd_raw = np.load(seq_pcds[i], allow_pickle=True)
            pcd_clean = np.nan_to_num(pcd_raw, nan=40.0, posinf=40.0, neginf=0.3)
            pcd_clean = colorize_depth(pcd_clean)
            pcd_crop = crop_matrix(pcd_clean, resize=config.scale, D3=False, crop=self.config.crop_roi)
            pcd_transpose = pcd_crop.transpose(2, 0, 1)
            data["pcd_xs"].append(torch.from_numpy(np.array(pcd_transpose[0:1, :, :])))
            data["pcd_zs"].append(torch.from_numpy(np.array(pcd_transpose[2:3, :, :])))

        ego_local_x = seq_local_xs[0]
        ego_local_y = seq_local_ys[0]
        ego_local_heading = seq_local_headings[0]
        # convert waypoint to local coordinate
        data["waypoints"] = []
        for j in range(1, self.pred_len + 1):
            local_waypoint = transform_2d_points(
                np.zeros((1, 3)),
                np.pi/2 - seq_local_headings[j], seq_local_xs[j], seq_local_ys[j],
                np.pi/2 - ego_local_heading, ego_local_x, ego_local_y
            )
            data["waypoints"].append(tuple(local_waypoint[0, :2]))
        # convert rp1, rp2 from gnss to local coordinates
        bearing_robot = self.bearing[index]
        lat_robot = self.lat[index]
        lon_robot = self.lon[index]
        R_matrix = np.array([
            [np.cos(bearing_robot), -np.sin(bearing_robot)],
            [np.sin(bearing_robot),  np.cos(bearing_robot)]
        ])
        dLat1_m = (self.rp1_lat[index] - lat_robot) * 40008000 / 360
        dLon1_m = (self.rp1_lon[index] - lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360
        dLat2_m = (self.rp2_lat[index] - lat_robot) * 40008000 / 360
        dLon2_m = (self.rp2_lon[index] - lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360
        data['rp1'] = tuple(R_matrix.T.dot(np.array([dLon1_m, dLat1_m])))
        data['rp2'] = tuple(R_matrix.T.dot(np.array([dLon2_m, dLat2_m])))

        data["velocity"] = self.velocity[index]

        return data
        

if __name__ == "__main__":
    dataset = KarrDataset(split='train')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4, drop_last=False)
    iterator = iter(dataloader)
    first_step = next(iter(iterator))
    print(first_step)

    # for batch in dataloader:
    #     print(batch)

            


