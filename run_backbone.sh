
for ms in 100 200 300 ;
    do
    for backbone in resnet18 resnet34 resnet50 ;
      do
      exp_name=backbone_${backbone}
      echo ${exp_name}
      python main_drift_etf.py  --exp_name ${exp_name} --drift_lamda  0.1  --lamda 1 --batch_size 512 --model ${backbone} --dataset cifar10 --mem_size ${ms} --epochs 100  --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
      python main_linear_buffer.py --dataset cifar10 --learning_rate 0.1 --target_task 4 --ckpt  ./save_random_200/cifar10_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
      python emailUtil.py --name ${exp_name}
    done

done