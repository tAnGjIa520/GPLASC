for local_lamda in 0.5 ;
do
  for distill_lamda in 0.05 ;
  do
    for seed in 1997 2001 2024 ;
    do
      exp_name=cifar10_distill_lamda_${distill_lamda}_local_lamda_${local_lamda}_relu_200_${seed}

      echo ${exp_name}
      python main_drift_etf.py --exp_name ${exp_name} --drift_lamda ${distill_lamda} --lamda ${local_lamda} --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100 --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01 --cosine;
      python main_linear_buffer.py --dataset cifar10 --learning_rate 1 --target_task 4 --ckpt ./save_random_200/cifar10_models/${exp_name} --logpt ./save_random_200/logs/${exp_name} &
      python emailUtil.py --name ${exp_name}
    done
  done
done
