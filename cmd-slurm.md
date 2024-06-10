sbatch tools/bme_train.sh  configs/simclr/simclr_resnet50_1xb64-coslr-400e_reflacx.py 

sbatch tools/bme_train.sh  configs/byol/byol_resnet50_1xb64-coslr-400e_reflacx.py 

sbatch tools/bme_train.sh  configs/barlowtwins/barlowtwins_resnet50_1xb64-coslr-400e_reflacx.py --resume work_dirs/barlowtwins_resnet50_1xb64-coslr-400e_reflacx/epoch_100.pth



