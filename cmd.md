bash tools/dist_train.sh configs/mocov2/mocov2_resnet50_1xb64-coslr-800e_reflacx.py 1

bash tools/dist_train.sh configs/simclr/simclr_resnet50_1xb64-coslr-400e_reflacx.py 1

bash tools/dist_train.sh configs/cae/cae_beit-base-p16_1xb64-amp-coslr-800e_reflacx.py 1
