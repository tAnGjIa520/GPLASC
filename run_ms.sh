for ms in 50 100 300 400 ;
    do
    exp_name=ms_${ms}
    echo ${exp_name}
    python main_drift_etf.py  --exp_name ${exp_name} --drift_lamda  0.1  --lamda 1 --batch_size 512 --model resnet18 --dataset cifar10 --mem_size ${ms} --epochs 100  --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
    python main_linear_buffer.py --dataset cifar10 --learning_rate 0.1 --target_task 4 --ckpt  ./save_random_200/cifar10_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
    python emailUtil.py --name ${exp_name}
  done


## 原来
#for ms in 50 100 300 400 ;
#    do
#    exp_name=ms_${ms}
#    echo ${exp_name}
#    python main_drift_etf.py --exp_name ${exp_name} --batch_size 512 --model resnet18 --dataset cifar10 --mem_size ${ms} --epochs 100  --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
#    python main_linear_buffer.py --dataset cifar10 --learning_rate 0.1 --target_task 4 --ckpt  ./save_random_200/cifar10_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
#    python emailUtil.py --name ${exp_name}
#  done