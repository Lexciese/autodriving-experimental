import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from pathlib import Path
import numpy as np
import yaml
from utility import latlon_to_yaw, euler_from_quaternion, transform_2d_points, resizecrop_matrix, crop_matrix, cls2one_hot
from config import GlobalConfig

config = GlobalConfig()

class KarrDataset(Dataset):
    def __init__(self, sequence_length=None, transform=None):
        self.config = config
        self.seq_len = self.config.seq_len
        self.pred_len = self.config.pred_len
        self.data_rate = self.config.data_rate
        self.root_dir = Path(self.config.root_dir)
        self.train_sequences = self.config.train_sequences
        self.val_sequences = self.config.val_sequences
        self.test_sequences = self.config.test_sequences

        self.filename = []
        self.rgb = []
        self.seg = []
        self.pointcloud = []
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
        self.velocity = []

        self.data = None

        # meta
        self.dir_meta = self.root_dir / "meta"

        # rgbd directory
        self.dir_rgb_front = self.root_dir / "camera" / "rgb"
        self.dir_rgb_seg = self.root_dir / "camera" / "seg" / "img"
        self.dir_depth_front = self.root_dir / "camera" / "depth" / "cld2"
        self.dir_histo = self.root_dir / "camera" / "histogram"
        self.dir_optflow = self.root_dir / "camera" / "optical_flow"

        self.rp_list = None
        preload_file = self.root_dir /  "preload.npy"
        if not os.path.exists(preload_file):
            preload_filename = []
            preload_rgb = []
            preload_seg = []
            preload_pointcloud = []
            preload_lon = []
            preload_lat = []
            preload_local_x = []
            preload_local_y = []
            preload_local_heading = []
            preload_rp1_lon = []
            preload_rp1_lat = []
            preload_rp2_lon = []
            preload_rp2_lat = []
            preload_bearing = []
            preload_velocity = []

            # load route
            with open(self.root_dir / "2026-02-09_route00_routepoint_list.yml", "r") as x:
                self.rp_list = yaml.safe_load(x)
                # assign end point sebagai route terakhir
                self.rp_list['route_point']['latitude'].append(self.rp_list['last_point']['latitude'])
                self.rp_list['route_point']['longitude'].append(self.rp_list['last_point']['longitude'])
            
            files = os.listdir(self.dir_meta)
            files.sort()
            files = [os.path.splitext(filename)[0] for filename in files] # remove extension string
            self.len_files = len(files)

            for i in range(0, len(files) - (self.seq_len-1) - (self.pred_len * self.data_rate)):
                # The sequence data used for the model
                rgbs = []
                segs = []
                pointclouds = []
                local_xs = []
                local_ys = []
                local_headings = []

                # read files sequentially (past and current frames)
                # t-2, t-1, t
                for j in range(0, self.seq_len):
                    filename = files[i+j]
                    rgbs.append(self.dir_rgb_front / f"{filename}.png")
                    segs.append(self.dir_rgb_seg / f"{filename}.png")
                    pointclouds.append(self.dir_depth_front / f"{filename}.npy")

                preload_filename.append(filename)
                preload_rgb.append(rgbs)
                preload_seg.append(segs)
                preload_pointcloud.append(pointclouds)
                
                # current meta
                with open(self.dir_meta / f"{filename}.yml", 'r') as curr_metafile:
                    current_meta = yaml.safe_load(curr_metafile)
                current_latitude = current_meta['global_position_latlon'][0]
                current_longitude = current_meta['global_position_latlon'][1]
                preload_lat.append(current_latitude)
                preload_lon.append(current_longitude)

                local_xs.append(current_meta['local_position_xyz'][0])
                local_ys.append(current_meta['local_position_xyz'][1])

                qx, qy, qz, qw = current_meta['global_orientation_xyzw']
                _, _, yaw = euler_from_quaternion(qw, qx, qy, qz)
                local_headings.append(yaw)

                preload_velocity.append(current_meta['velocity'])

                # ===================================================

                with open(self.dir_meta / f"{files[i-1]}.yml", 'r') as prev_metafile:
                    prev_meta = yaml.safe_load(prev_metafile)
                veh_prev_lat = prev_meta['global_position_latlon'][0]
                veh_prev_lon = prev_meta['global_position_latlon'][1]

                #SUDAH GA DIPAKE, IMU NOISY
                bearing_veh = latlon_to_yaw(current_latitude, current_longitude, veh_prev_lat, veh_prev_lon, offset=np.pi / 2.0)
                bearing_veh_deg = np.degrees(bearing_veh) - 90
                bearing_veh = np.radians(bearing_veh_deg)

                # get velocity
                velocity =  current_meta['velocity']

                R_matrix = np.array([[np.cos(bearing_veh), -np.sin(bearing_veh)],
                                    [np.sin(bearing_veh),  np.cos(bearing_veh)]])
                about_to_finish = False
                for j in range(2): #ada 2 route point
                    next_lat = self.rp_list['route_point']['latitude'][j]
                    next_lon = self.rp_list['route_point']['longitude'][j]
                    
                    dLat_m = (next_lat - current_latitude) * 40008000 / 360 #111320 #Y
                    dLon_m = (next_lon - current_longitude) * 40075000 * np.cos(np.radians(current_longitude)) / 360 #X

                    # If the euclidian distance of rp1 <= min_distance, remove the route then skip to next route
                    if j == 0 and np.sqrt(dLat_m**2 + dLon_m**2) <= self.config.rp1_close and not about_to_finish: 
                        # If the route list count > 2
                        if len(self.rp_list['route_point']['latitude']) > 2:
                            self.rp_list['route_point']['latitude'].pop(0)
                            self.rp_list['route_point']['longitude'].pop(0)
                        # Near finish
                        else:
                            about_to_finish = True
                            self.rp_list['route_point']['latitude'][0] = self.rp_list['route_point']['latitude'][-1]
                            self.rp_list['route_point']['longitude'][0] = self.rp_list['route_point']['longitude'][-1]

                        next_lat = self.rp_list['route_point']['latitude'][j]
                        next_lon = self.rp_list['route_point']['longitude'][j]

                    if j == 0:
                        preload_rp1_lon.append(next_lon)
                        preload_rp1_lat.append(next_lat)
                    elif j == 1:
                        preload_rp2_lon.append(next_lon)
                        preload_rp2_lat.append(next_lat)
                
                # read files sequentially (future frames)
                for k in range(1, self.pred_len + 1):
                    file = files[(i + self.seq_len - 1) + (k*self.data_rate)]
                    # meta
                    with open(self.dir_meta / f"{file}.yml", "r") as read_future_meta:
                        future_meta = yaml.load(read_future_meta, Loader=yaml.FullLoader)
                    local_xs.append(future_meta["local_position_xyz"][0])
                    local_ys.append(future_meta["local_position_xyz"][1])
                    fqx, fqy, fqz, fqw = future_meta['global_orientation_xyzw']
                    future_yaw =  euler_from_quaternion(fqw, fqx, fqy, fqz)
                    local_headings.append(future_yaw)

                preload_local_x.append(local_xs)
                preload_local_y.append(local_ys)
                preload_local_heading.append(local_headings)

                # print("current:", current_meta["local_position_xyz"][0])
                # print("future: ", future_meta["local_position_xyz"][0])
                # print("==================")


            # dump to npy
            preload_dict = {}
            preload_dict["filename"] = preload_filename 
            preload_dict["rgb"] = preload_rgb 
            preload_dict["seg"] = preload_seg 
            preload_dict["pointcloud"] = preload_pointcloud 
            preload_dict["lon"] = preload_lon 
            preload_dict["lat"] = preload_lat 
            preload_dict["local_x"] = preload_local_x 
            preload_dict["local_y"] = preload_local_y 
            preload_dict["local_heading"] = preload_local_heading 
            preload_dict["rp1_lon"] = preload_rp1_lon 
            preload_dict["rp1_lat"] = preload_rp1_lat 
            preload_dict["rp2_lon"] = preload_rp2_lon 
            preload_dict["rp2_lat"] = preload_rp2_lat 
            preload_dict["bearing"] = preload_bearing 
            preload_dict["velocity"] = preload_velocity 
            np.save(preload_file, preload_dict)


        # load from npy if available
        preload_dict = np.load(preload_file, allow_pickle=True)
        self.filename += preload_dict.item()["filename"]
        self.rgb += preload_dict.item()["rgb"]
        self.seg += preload_dict.item()["seg"]
        self.pointcloud += preload_dict.item()["pointcloud"]
        self.lon += preload_dict.item()["lon"]
        self.lat += preload_dict.item()["lat"]
        self.local_x += preload_dict.item()["local_x"]
        self.local_y += preload_dict.item()["local_y"]
        self.local_heading += preload_dict.item()["local_heading"]
        self.rp1_lon += preload_dict.item()["rp1_lon"]
        self.rp1_lat += preload_dict.item()["rp1_lat"]
        self.rp2_lon += preload_dict.item()["rp2_lon"]
        self.rp2_lat += preload_dict.item()["rp2_lat"]
        self.bearing += preload_dict.item()["bearing"]
        self.velocity += preload_dict.item()["velocity"]
        print("Preloading dataset")
    
    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, index):
        print(index)
        data = dict()
        data['rgbs'] = []
        data['segs'] = []
        seq_rgbs = self.rgb[index]
        seq_segs = self.seg[index]
        data['pointcloud_xs'] = []
        data['pointcloud_zs'] = []
        seq_pointcloud = self.pointcloud[index]
        seq_local_xs = self.local_x[index]
        seq_local_ys = self.local_y[index]
        seq_local_headings = self.local_heading[index]

        for i in range(0, self.seq_len):
            data['rgbs'].append(torch.from_numpy(np.array(crop_matrix(cv2.imread(seq_rgbs[i]), resize=self.config.scale, crop=self.config.crop_roi).transpose(2, 0, 1))))
            data['segs'].append(torch.from_numpy(np.array(cls2one_hot(crop_matrix(cv2.imread(seq_segs[i]), resize=self.config.scale, crop=self.config.crop_roi), n_class=self.config.n_class))))
            pcd = np.nan_to_num(crop_matrix(np.load(seq_pointcloud[i], allow_pickle=True)[:,:,0:3], resize=self.config.scale, crop=self.config.crop_roi).transpose(2,0,1), nan=0.0, posinf=39.99999, neginf=0.2) #min_d, max_d, -max_d, ambil xyz-nya saja 0:3
            data['pointcloud_xs'].append(torch.from_numpy(np.array(pcd[0:1,:,:])))
            data['pointcloud_zs'].append(torch.from_numpy(np.array(pcd[2:3,:,:])))

        ego_local_x = seq_local_xs[0]
        ego_local_y = seq_local_ys[0]
        ego_local_heading = seq_local_headings[0]

        # waypoint processing to local coordinates
        data['waypoints'] = [] # wp dalam local coordinate
        # for j in range(1, self.pred_len + 1):
        #     local_waypoint = transform_2d_points(np.zeros((1,3)), np.pi/2-seq_local_headings[j], seq_local_xs[j], seq_local_ys[j], np.pi/2-ego_local_heading, ego_local_x, ego_local_y)
        #     data['waypoints'].append(tuple(local_waypoint[0:2]))

        # convert rp1_lon, rp1_lat rp2_lon, rp2_lat ke local coordinates
        # compute from glocal to local
        bearing_robot = self.bearing[index]
        lat_robot = self.lat[index]
        lon_robot = self.lon[index]
        R_matrix = np.array([[np.cos(bearing_robot), -np.sin(bearing_robot)],
                            [np.sin(bearing_robot),  np.cos(bearing_robot)]])
        dLat1_m = (self.rp1_lat[index]-lat_robot) * 40008000 / 360 #111320 #Y
        dLon1_m = (self.rp1_lon[index]-lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360 #X
        dLat2_m = (self.rp2_lat[index]-lat_robot) * 40008000 / 360 #111320 #Y
        dLon2_m = (self.rp2_lon[index]-lon_robot) * 40075000 * np.cos(np.radians(lat_robot)) / 360 #X
        data['rp1'] = tuple(R_matrix.T.dot(np.array([dLon1_m, dLat1_m])))
        data['rp2'] = tuple(R_matrix.T.dot(np.array([dLon2_m, dLat2_m])))

        data['velocity'] = tuple(np.array([self.velocity[index]]))
        data['bearing_robot'] = np.degrees(bearing_robot)
        data['lat_robot'] = lat_robot
        data['lon_robot'] = lon_robot

        return data

class KARR_DataModule(pl.LightningDataModule):
    def __init__(self, sequences_length, config):
        super().__init__()
        self.sequences_length = sequences_length
        self.config = config.GlobalConfig
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = KarrDataset(
                sequence_length=self.sequences_length
            )
            print(f"Train len: {len(self.train_dataset)}")

        if stage == "validate" or stage == "fit" or stage is None:
            self.val_dataset = KarrDataset(
                sequence_length=self.sequences_length
            )
            print(f"Val len: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = KarrDataset(
                sequence_length=self.sequences_length
            )
            print(f"Test len: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True
        )
        print(f"Total training samples: {len(loader.dataset)}")
        return loader
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=False
        )

if __name__ == "__main__":
    KarrDataset()


  