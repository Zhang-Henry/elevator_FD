# CUDA_VISIBLE_DEVICES=4 nohup python FD.py --bz 1024 --gamma 2 --alpha 0.25 0.25 0.35 0.45 0.7 0.7 \
#     > logs/FD_imb_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python FD.py --bz 1024 --gamma 2 --alpha 0.25 0.6 0.6 0.6 0.4 0.4  \
    > logs/FD_imb_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python FD_scl_train.py > logs/FD_scl_train.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python FD.py > logs/FD_imb.log 2>&1 &