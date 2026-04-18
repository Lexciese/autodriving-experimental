import os
import time
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg_module
from model_ikaz import ai23
import dataloader as dataloader_module
from train import prepare_batch
from utility import crop_matrix

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
# Each panel matches the natural image resolution after crop+scale: 512 wide x 256 tall
PANEL_W = 512
PANEL_H = 256
FRAME_WIDTH  = PANEL_W * 2   # 1024
FRAME_HEIGHT = PANEL_H * 2   # 512
OUTPUT_FILE  = "ai23_evaluation_video_bev.mp4"
MAX_FRAMES   = 500

# ZED 2i estimated intrinsics after crop_roi=[512,1024] + scale=2
# Original 1280x720 → crop to 1024x512 (center) → scale=2 → 512x256
# cx = center of 512-wide image = 256, fx estimated from ZED2i ~527/2 ≈ 263, use 400 as default
# Increase FX_EST → narrower fan; decrease → wider fan
FX_EST = 400
CX_EST = 256   # principal point x (center of 512-wide image)


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def tensor_to_bgr(tensor_img):
    """Convert a (3, H, W) raw BGR tensor (values 0-255 float) to (H, W, 3) uint8 BGR.
    The dataloader loads via cv2.imread (BGR) and stores as float without normalization,
    so no mean/std un-doing is needed — just clip and transpose."""
    img = tensor_img.cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))   # (H, W, 3) BGR


def meter_to_bev_pixel(x_meter, z_meter, w, h, cover_area):
    px = int(round((x_meter + cover_area) * (w - 1) / (2 * cover_area)))
    pz = int(round((z_meter * (1 - h) / cover_area) + (h - 1)))
    return px, pz


def clamp_pixel(px, pz, w, h):
    return max(0, min(px, w - 1)), max(0, min(pz, h - 1))


# --------------------------------------------------------------
# Panel renderers  (each outputs PANEL_H x PANEL_W = 256x512 BGR)
# --------------------------------------------------------------
def render_rgb_panel(rgb_tensor):
    """Top-left: RGB front camera → 256x512 BGR"""
    img_bgr = tensor_to_bgr(rgb_tensor)                              # (256, 512, 3)
    return cv2.resize(img_bgr, (PANEL_W, PANEL_H))


def render_depth_vis_panel(depth_path, cfg):
    """Top-right: Depth map grayscale (close=bright, far=dark) → 256x512 BGR"""
    depth_raw = np.load(depth_path, allow_pickle=True)
    depth_raw = np.nan_to_num(depth_raw, nan=40.0, posinf=40.0, neginf=0.3)
    depth_crop = crop_matrix(depth_raw, resize=cfg.scale, D3=False, crop=cfg.crop_roi)  # (256, 512)
    vis = -depth_crop / 10.0 + 1
    vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    return cv2.resize(vis_bgr, (PANEL_W, PANEL_H))


def render_seg_panel(seg_tensor, colors_bgr):
    """Bottom-left: GT semantic segmentation colorized → 256x512 BGR"""
    seg_idx     = torch.argmax(seg_tensor, dim=0).cpu().numpy()     # (256, 512)
    seg_colored = colors_bgr[seg_idx]                               # (256, 512, 3)
    return cv2.resize(seg_colored, (PANEL_W, PANEL_H))


def render_depth_bev_panel(depth_path, seg_path, pred_wp, gt_wp, rp1, rp2, cfg, colors_bgr):
    """Bottom-right: Depth point cloud top-down BEV (colored by seg class)
    + predicted waypoints (red) + route points (blue Rp1, yellow Rp2) → 256x512 BGR

    BEV canvas is PANEL_W x PANEL_H (512 x 256):
      - width  covers ±coverage_area m laterally  (wider view)
      - height covers  0..coverage_area m forward
    """
    BW, BH = PANEL_W, PANEL_H   # BEV canvas dimensions

    # --- Depth point cloud BEV (vectorized) ---
    depth_raw = np.load(depth_path, allow_pickle=True)
    depth_raw = np.nan_to_num(depth_raw, nan=40.0, posinf=40.0, neginf=0.3)
    depth_crop = crop_matrix(depth_raw, resize=cfg.scale, D3=False, crop=cfg.crop_roi)  # (256, 512)

    seg_bgr   = cv2.imread(seg_path)
    seg_crop  = crop_matrix(seg_bgr, resize=cfg.scale, D3=True, crop=cfg.crop_roi)      # (256, 512, 3)
    class_idx = seg_crop[:, :, 0]                                                        # (256, 512)

    H, W = depth_crop.shape
    uu   = np.tile(np.arange(W), (H, 1))                            # (H, W) column indices
    valid = (depth_crop > 0.3) & (depth_crop < cfg.coverage_area)

    d   = depth_crop[valid]
    u   = uu[valid]
    cls = class_idx[valid]

    # Pinhole projection → BEV (X=lateral, Z=forward)
    x_m = d * (u - CX_EST) / FX_EST
    z_m = d

    # Map to BEV canvas pixels using the BEV canvas dimensions
    bev_u = np.round((x_m + cfg.coverage_area) * (BW - 1) / (2 * cfg.coverage_area)).astype(int)
    bev_v = np.round((z_m * (1 - BH) / cfg.coverage_area) + (BH - 1)).astype(int)

    in_bounds = (bev_u >= 0) & (bev_u < BW) & (bev_v >= 0) & (bev_v < BH)

    canvas = np.zeros((BH, BW, 3), dtype=np.uint8)
    canvas[bev_v[in_bounds], bev_u[in_bounds]] = colors_bgr[cls[in_bounds]]

    # --- Ego vehicle (white dot) ---
    ego_px, ego_pz = clamp_pixel(*meter_to_bev_pixel(0, 0, BW, BH, cfg.coverage_area), BW, BH)
    cv2.circle(canvas, (ego_px, ego_pz), 5, (255, 255, 255), -1)

    # --- Route point 1 (blue), Route point 2 (yellow) ---
    for rp_vec, color in [(rp1, (255, 0, 0)), (rp2, (0, 255, 255))]:
        px, pz = clamp_pixel(*meter_to_bev_pixel(rp_vec[0], rp_vec[1], BW, BH, cfg.coverage_area), BW, BH)
        cv2.circle(canvas, (px, pz), 6, color, -1)

    # --- GT waypoints (green line + dots) ---
    prev = (ego_px, ego_pz)
    for i in range(cfg.pred_len):
        px, pz = clamp_pixel(*meter_to_bev_pixel(gt_wp[i][0], gt_wp[i][1], BW, BH, cfg.coverage_area), BW, BH)
        cv2.line(canvas, prev, (px, pz), (0, 255, 0), 2)
        cv2.circle(canvas, (px, pz), 3, (0, 255, 0), -1)
        prev = (px, pz)

    # --- Predicted waypoints (red line + dots) ---
    prev = (ego_px, ego_pz)
    for i in range(cfg.pred_len):
        px, pz = clamp_pixel(*meter_to_bev_pixel(pred_wp[i][0], pred_wp[i][1], BW, BH, cfg.coverage_area), BW, BH)
        print(pred_wp[1, 0])
        cv2.line(canvas, prev, (px, pz), (0, 0, 255), 2)
        cv2.circle(canvas, (px, pz), 3, (0, 0, 255), -1)
        prev = (px, pz)

    return canvas


def assemble_frame(q_tl, q_tr, q_bl, q_br):
    top = np.hstack((q_tl, q_tr))   # (256, 1024, 3)
    bot = np.hstack((q_bl, q_br))   # (256, 1024, 3)
    return np.vstack((top, bot))     # (512, 1024, 3)


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------
def main():
    cfg    = cfg_module.GlobalConfig()
    device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading test dataset...")
    val_ds     = dataloader_module.KarrDataset(split='val')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    net = ai23(config=cfg, device=device).to(device)
    checkpoint_path = os.path.join("/media/mf/AUTODRIVING-4TB/ai23/log/ai23_21_47_12", 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at {checkpoint_path}")
        return
    print(f"Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt)
    net.eval()

    # Color table: RGB → BGR for OpenCV
    colors_bgr = np.array([[c[2], c[1], c[0]] for c in cfg.SEG_CLASSES['colors']], dtype=np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(OUTPUT_FILE, fourcc, 5.0, (FRAME_WIDTH, FRAME_HEIGHT))

    print("\nRendering BEV video...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, total=min(len(val_loader), MAX_FRAMES))):
            if batch_idx >= MAX_FRAMES:
                break

            rgbs, segs, pcd_xs, pcd_zs, rp1, rp2, velocity, gt_wp = prepare_batch(batch, device, cfg)
            pred_segs, pred_wp_meter, sdcs = net(rgbs, pcd_xs, pcd_zs, rp1, rp2, velocity)

            depth_path = val_ds.pcd[batch_idx][-1]
            seg_path   = val_ds.seg[batch_idx][-1]

            # Panel 1 (top-left): RGB camera
            q_tl = render_rgb_panel(rgbs[-1][0])
            speed_kmh = velocity[0].item() * 30
            cv2.putText(q_tl, f"Kecepatan: {speed_kmh:.1f} km/h", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Panel 2 (top-right): Depth grayscale
            q_tr = render_depth_vis_panel(depth_path, cfg)

            # Panel 3 (bottom-left): GT segmentation
            q_bl = render_seg_panel(segs[-1][0], colors_bgr)

            # Panel 4 (bottom-right): Depth BEV + predicted WP + route points
            q_br = render_depth_bev_panel(
                depth_path, seg_path,
                pred_wp_meter[0].cpu().numpy(),   # [pred_len, 2] predicted
                gt_wp[0].cpu().numpy(),           # [pred_len, 2] ground truth
                rp1[0].cpu().numpy(),             # [2] meters
                rp2[0].cpu().numpy(),             # [2] meters
                cfg, colors_bgr
            )

            frame = assemble_frame(q_tl, q_tr, q_bl, q_br)
            out.write(frame)

    out.release()
    print(f"\nBEV video saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
