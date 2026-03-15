import numpy as np
import cv2
import torch
import torch.nn.functional as F

def swap_RGB2BGR(matrix):
    red = matrix[:,:,0].copy()
    blue = matrix[:,:,2].copy()
    matrix[:,:,0] = blue
    matrix[:,:,2] = red
    return matrix

def euler_from_quaternion(w, x, y, z, rad=True): #urutannya q0, q1, q2, q3
    #https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    #https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    if rad:
        return roll_x, pitch_y, yaw_z # in radians
    else:
        return np.degrees(roll_x), np.degrees(pitch_y), np.degrees(yaw_z)

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out

def latlon_to_yaw(lat, lon, lat0, lon0, offset=0.0):
    lat, lon, lat0, lon0 = map(np.radians, [lat, lon, lat0, lon0])
    dlon = lon - lon0
    x = np.sin(dlon) * np.cos(lat)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    yaw = np.arctan2(-x, y)
    return ((yaw + offset) + np.pi) % (2 * np.pi) - np.pi

def quaternion_to_yaw(quat: list, offset=0.0):
    yaw_list = []
    for q in quat:
        w, x, y, z = q[3], q[0], q[1], q[2]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*2 + z*2))
        yaw_list.append(((yaw + offset) + np.pi) % (2 * np.pi) - np.pi)
    return yaw_list

def colorize_depth(depth_map):
    norm_dep = depth_map / 10.0 #diubah terjauh 1, terdekat 0 karena dari ros2 message, max 9.99999, min 0.3
    norm_dep = -1*norm_dep + 1 #dibalik terjauh 0, terdekat 1
    visdep = np.repeat(norm_dep[:, :, np.newaxis], 3, axis=2) * 255 #normalisasi ke 0 - 255
    return visdep

def resizecrop_matrix(image, WH_resized=[256, 128], D3=True, crop_HW=[128, 256]):

    #resize image
    resized_image = cv2.resize(image, WH_resized, interpolation=cv2.INTER_NEAREST)

    # print(image.shape)
    # upper_left_yx = [int((image.shape[0]/2) - (crop/2)), int((image.shape[1]/2) - (crop/2))]
    upper_left_yx = [int((resized_image.shape[0]/2) - (crop_HW[0]/2)), int((resized_image.shape[1]/2) - (crop_HW[1]/2))]
    if D3: #buat matrix 3d
        cropped_im = resized_image[upper_left_yx[0]:upper_left_yx[0]+crop_HW[0], upper_left_yx[1]:upper_left_yx[1]+crop_HW[1], :]
    else: #buat matrix 2d
        cropped_im = resized_image[upper_left_yx[0]:upper_left_yx[0]+crop_HW[0], upper_left_yx[1]:upper_left_yx[1]+crop_HW[1]]


    return cropped_im

def crop_matrix(image, resize=1, D3=True, crop=[512, 1024]):
    
    # print(image.shape)
    # upper_left_yx = [int((image.shape[0]/2) - (crop/2)), int((image.shape[1]/2) - (crop/2))]
    upper_left_yx = [int((image.shape[0]/2) - (crop[0]/2)), int((image.shape[1]/2) - (crop[1]/2))]
    if D3: #buat matrix 3d
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1], :]
    else: #buat matrix 2d
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1]]

    #resize image
    WH_resized = (int(cropped_im.shape[1]/resize), int(cropped_im.shape[0]/resize))
    resized_image = cv2.resize(cropped_im, WH_resized, interpolation=cv2.INTER_NEAREST)

    return resized_image

def cls2one_hot(ss_gt, n_class):
    #inputnya adalah HWC baca cv2 secara biasanya, ambil salah satu channel saja
    ss_gt = np.transpose(ss_gt, (2,0,1)) #GANTI CHANNEL FIRST
    ss_gt = ss_gt[:1,:,:].reshape(ss_gt.shape[1], ss_gt.shape[2])
    result = (np.arange(n_class) == ss_gt[...,None]).astype(int) # jumlah class di cityscape pallete
    result = np.transpose(result, (2, 0, 1))   # (H, W, C) --> (C, H, W)
    # np.save("00009_ss.npy", result) #SUDAH BENAR!
    # print(result)
    # print(result.shape)
    return result


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out

#buat ngecek GT SEG aja
def check_gt_seg(config, gt_seg):
    gt_seg = gt_seg.cpu().detach().numpy()

    #buat array untuk nyimpan out gambar
    imgx = np.zeros((gt_seg.shape[2], gt_seg.shape[3], 3))
    #ambil tensor segmentationnya
    inx = np.argmax(gt_seg[0], axis=0)
    for cmap in config.SEG_CLASSES['colors']:
        cmap_id = config.SEG_CLASSES['colors'].index(cmap)
        imgx[np.where(inx == cmap_id)] = cmap

    #GANTI ORDER BGR KE RGB, SWAP!
    imgx = swap_RGB2BGR(imgx)
    cv2.imwrite(config.logdir+"/check_gt_seg.png", imgx) #cetak gt segmentation


#Class untuk penyimpanan dan perhitungan update loss
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    #update kalkulasi
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#Class NN Module untuk Perhitungan BCE Dice Loss
def BCEDice(Yp, Yt, smooth=1e-7):
    #.view(-1) artinya matrix tensornya di flatten kan dulu
    Yp = Yp.view(-1)
    Yt = Yt.view(-1)
    #hitung BCE
    bce = F.binary_cross_entropy(Yp, Yt, reduction='mean')
    #hitung dice loss
    intersection = (Yp * Yt).sum() #irisan
    #rumus DICE
    dice_loss = 1 - ((2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth))
    #kalkulasi lossnya
    bce_dice_loss = bce + dice_loss
    return bce_dice_loss


#fungsi renormalize loss weights seperti di paper gradnorm
def renormalize_params_lw(current_lw, config):
    #detach dulu paramsnya dari torch, pindah ke CPU
    lw = np.array([tens.cpu().detach().numpy() for tens in current_lw])
    lws = np.array([lw[i][0] for i in range(len(lw))])
    #fungsi renormalize untuk algoritma 1 di papaer gradnorm
    coef = np.array(config.loss_weights).sum()/lws.sum()
    new_lws = [coef*lwx for lwx in lws]
    #buat torch float tensor lagi dan masukkan ke cuda memory
    normalized_lws = [torch.cuda.FloatTensor([lw]).clone().detach().requires_grad_(True) for lw in new_lws]
    return normalized_lws