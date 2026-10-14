
  for threshhold in 0 0.1 0.2 0.3 ;
    do
    exp_name=${threshhold}_thresh
    echo ${exp_name}
    python main_drift_etf_threshhold.py --exp_name ${exp_name} --threshhold ${threshhold} --threshhold ${threshhold} --drift_lamda 0.05 --lamda 1 --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --start_epoch 500 --epochs 100  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
    python main_linear_buffer.py --dataset cifar10 --learning_rate 0.5 --target_task 4 --ckpt  ./save_random_200/cifar10_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
    python emailUtil.py --name ${exp_name}
  done

#
#
#只有对内积的约束，没有分开

# 带有mlp头的
#IRD+AE
#for distill_lamda in 0.5 0.1 0.05;
#  do
#  exp_name=ird_ae_${distill_lamda}
#  echo ${exp_name}
#  python main_drift_etf_mlp.py --exp_name  ${exp_name} --drift_lamda ${distill_lamda} --lamda 0 --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100  --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
#  python main_linear_buffer.py --dataset cifar10 --learning_rate 0.3 --target_task 4 --ckpt  ./save_random_200/cifar10_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
#  python emailUtil.py --name ${exp_name}
#done

##IRD+MSE
#for distill_lamda in 0.5 0.1 0.05;
#    do
#    exp_name=ird_mse_${distill_lamda}
#    echo ${exp_name}
#    python main_drift_etf_mse.py --exp_name ${exp_name} --drift_lamda ${distill_lamda} --lamda 0 --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100  --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
#    python main_linear_buffer.py --dataset cifar10 --learning_rate 0.3 --target_task 4 --ckpt  ./save_random_200/cifar10_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
#    python emailUtil.py --name ${exp_name}
#  done
