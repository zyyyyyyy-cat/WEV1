# DDP
## WHU-MARS-1000

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/SCGPD_UAD.yml \
  MODEL.DIST_TRAIN True \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR logs/SCGPD_1000_DDP

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/SCGPD_UAD.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  TEST.WEIGHT logs/SCGPD_1000_DDP/transformer_120.pth \
  TEST.DETAILED_DOMAIN_EVAL True \
  OUTPUT_DIR logs/SCGPD_1000_domain


## WHU-MARS-1000-GD

CUDA_VISIBLE_DEVICES=0 python test.py \
  --config_file configs/SCGPD_UAD.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth \
  DATASETS.ROOT_DIR /project/code/data \
  DATASETS.DATASET_DIR WHU-MARS \
  DATASETS.FILTER_GD True \
  TEST.IMS_PER_BATCH 256 \
  TEST.WEIGHT logs/SCGPD_1000_DDP/transformer_120.pth \
  OUTPUT_DIR logs/SCGPD_1000GD
