# ASCFormer/configs/segformer/segformer_mit-b2_rtm_1024x1024.py

_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/default_runtime.py",
]

# -----------------------
# Dataset / Paths (RTM)
# -----------------------
dataset_type = "RTMDataset"
data_root = "/media/general_storage6/rmastorage/datasets/RealTextManipulation/"   # <-- aponta para a tua pasta RTM (pode ser caminho absoluto)

# Ajusta estas pastas se tiveres outros nomes:
img_dir = "JPEGImages"
ann_dir = "SegmentationClass"

train_split = "train_v2.txt"
val_split  = "val_v2.txt"
test_split= "test_v2.txt"

crop_size = (1024, 1024)

# -----------------------
# Pipelines (iguais ao estilo Cityscapes, mas para RTM)
# -----------------------
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(2048, 1024), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

# -----------------------
# Dataloaders
# -----------------------
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        ann_file=train_split,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        ann_file=val_split,     # para já usamos test como val (se tiveres val.txt, mete aqui)
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        ann_file=test_split,    # ✅ TEST REAL
        pipeline=test_pipeline,
    ),
)

# -----------------------
# Evaluator (mIoU / mFscore do mmseg)
# -----------------------
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mFscore"])
test_evaluator = val_evaluator

# -----------------------
# Model: muda B0 -> B2 e muda head para 2 classes
# -----------------------
model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint="pretrain/mit_b2.pth"),
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=2,  # background vs tamper
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(768, 768)),
)

# -----------------------
# Optimizer (como no teu mit-b0_160k, AdamW + paramwise)
# -----------------------
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

# -----------------------
# LR schedule (Linear warmup + Poly)
# -----------------------
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=1500, end=160000, by_epoch=False),
]

# -----------------------
# Train loop (iter-based 160k)
# -----------------------
train_cfg = dict(type="IterBasedTrainLoop", max_iters=160000, val_interval=16000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")