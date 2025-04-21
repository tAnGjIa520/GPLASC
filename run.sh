for local_lamda in 1;
do
 for distill_lamda in 1 0.5 0.1;
  do
    for margin in 0;
    do
        for seed in 1997;
        do
           exp_name=cifar10_${distill_lamda}_${local_lamda}_v2_${margin}_500
           python main_drift_etf_v2.py --exp_name ${exp_name} --margin ${margin} --seed ${seed} --drift_lamda ${distill_lamda} --lamda ${local_lamda}  --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 500 --epochs 100 --start_epoch 500  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
           python main_linear_buffer.py --dataset cifar10 --learning_rate 0.1 --target_task 4 --ckpt  ./save_random_200/cifar10_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
           python emailUtil.py --name ${exp_name}
        done
    done
 done
done
