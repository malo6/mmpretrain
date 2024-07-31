sbatch tools/bme_train.sh  configs/simclr/simclr_resnet50_1xb64-coslr-400e_reflacx.py 

sbatch tools/bme_train.sh  configs/byol/byol_resnet50_1xb64-coslr-400e_reflacx.py 

sbatch tools/bme_train.sh  configs/barlowtwins/barlowtwins_resnet50_1xb64-coslr-400e_reflacx.py --resume work_dirs/barlowtwins_resnet50_1xb64-coslr-400e_reflacx/epoch_100.pth

sbatch tools/bme_train.sh  configs/mae/mae_vit-base-p16_1xb64-amp-coslr-400e_reflacx.py 

sbatch tools/bme_train.sh configs/mae/mae_vit-base-p16_1xb64-amp-coslr-400e_reflacx.py --work-dir workspace-timm-mask0.8

sbatch tools/bme_train.sh configs/mae/mae_vit-base-p16_1xb64-amp-coslr-400e_reflacx.py --work-dir workspace-mask0.8/0.5

sbatch tools/bme_train.sh configs/beit/beit_beit-base-p16_4xb64-amp-coslr-300e_reflacx.py --work-dir workspace-beit02

sbatch tools/bme_train.sh configs/cae/cae_beit-base-p16_4xb64-amp-coslr-300e_reflacx.py --work-dir workspace-cae01

sbatch tools/bme_train.sh configs/simmim/simmim_swin-base-w6_4xb64-amp-coslr-400e_reflacx-192px.py --work-dir workspace-simmim00


#beit版本使用的相对位置编码不匹配
sbatch tools/bme_train.sh configs/beitv2/beitv2_beit-base-p16_4xb64-amp-coslr-400e_reflacx.py --work-dir workspace-beitv2