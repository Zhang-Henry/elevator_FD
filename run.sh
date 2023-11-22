# CUDA_VISIBLE_DEVICES=7 nohup python FD.py > logs/FD_imb_$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python FD_scl_train.py > logs/FD_scl_train.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python FD.py > logs/FD_imb.log 2>&1 &