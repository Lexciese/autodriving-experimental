import os

class GlobalConfig:
    gpu_id = '0'
    num_workers = 4
    model = 'ai23'
    logdir = 'log/'+model+'_mix_mix'
    init_stop_counter = 30

    batch_size = 4 
    coverage_area = 30 # untuk top view SC, 24m kedepan, kiri, dan kanan
    rp1_close = 1 # ganti rp jika mendekati ...meter
    bearing_bias = 7.5 # dalam derajat, pastikan sama dengan yang ada di plot_wprp.py
    n_buffer = 5 # buffer untuk MAF dalam second
    data_rate = 5 # 1 detik ada berapa data?

    # Data
    seq_len = 3 # jumlah input seq
    pred_len = 3 # future waypoints predicted
    logdir = logdir+"_seq"+str(seq_len) # update direktori name


    # root_dir = '/media/mf/AUTODRIVING-4TB/ringroad/datasetx/2026-02-26_route00'
    root_dir = '/media/mf/AUTODRIVING-4TB/ringroad/datasetx/2026-02-26_route00'
    train_sequences = ["sequence01", "sequence02"]
    val_sequences = ["sequence03"]
    test_sequences = ["sequence04"]
    
    crop_roi = [512, 1024] #HxW
    scale = 2 # buat resizing diawal load data

    lr = 1e-5 # learning rate, pakai AdamW
    weight_decay = 1e-3

    #HANYA ADA 19 CLASS?? + #tambahan 0,0,0 hitam untuk area kosong pada SDC nantinya
    SEG_CLASSES = {
        'colors'        :[[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],  
                        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], 
                        [0, 80, 100], [0, 0, 230], [119, 11, 32]],  
        'classes'       : ['None', 'road', 'sidewalk', 'building', 'wall',
                            'fence', 'pole', 'traffic light', 'traffic sign', 
                            'vegetation', 'terrain', 'sky', 'person', 
                            'rider', 'car', 'truck', 'bus',
                            'train', 'motorcycle', 'bicycle']
    }
    n_class = len(SEG_CLASSES['colors'])

    n_fmap_b0 = [[32,16], [24], [40], [80,112], [192,320,1280]]
    n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,1280]] # sama dengan b0
    n_fmap_b2 = [[32,16], [24], [48], [88,120], [208,352,1408]]
    n_fmap_b3 = [[40,24], [32], [48], [96,136], [232,384,1536]] # lihat underdevelopment/efficientnet.py
    n_fmap_b4 = [[48,24], [32], [56], [112,160], [272,448,1792]]

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
