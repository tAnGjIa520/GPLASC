for threshhold in -1  0 0.1;
    do
    exp_name=single_threshhold_${threshhold}
    echo ${exp_name}
    python main_drift_etf_single_task_threshhold.py --threshhold ${threshhold} --exp_name ${exp_name} --drift_lamda  0  --lamda 1 --batch_size 512 --model resnet18 --dataset tiny-imagenet --mem_size 200 --epochs 100  --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
    python main_linear_buffer.py --dataset tiny-imagenet --learning_rate 0.1 --target_task 0 --ckpt  ./save_random_200/tiny-imagenet_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
    python emailUtil.py --name ${exp_name}
  done



for threshhold in 0.3 0.5 0.7;
    do
    exp_name=single_threshhold_${threshhold}
    echo ${exp_name}
    python main_drift_etf_single_task_threshhold.py --threshhold ${threshhold} --exp_name ${exp_name} --drift_lamda  0  --lamda 1 --batch_size 512 --model resnet18 --dataset tiny-imagenet --mem_size 200 --epochs 100  --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;

    python main_linear_buffer.py --dataset tiny-imagenet --learning_rate 0.1 --target_task 0 --ckpt  ./save_random_200/tiny-imagenet_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
    python emailUtil.py --name ${exp_name}
done
