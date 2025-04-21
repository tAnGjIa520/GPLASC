

##0.1 0.5 1
#for i in 0.005 0.01 0.015 0.02;
#  do echo $i;
#  python main_drift_etf.py --exp_name cifar10_distill_lamda_0.1_local_lamda_1_margin_$i --margin $i --drift_lamda  0.1 --lamda $i --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
#  python main_linear_buffer.py --dataset cifar10 --learning_rate 0.1 --target_task 4 --ckpt  ./save_random_200/cifar10_models/cifar10_distill_lamda_0.1_local_lamda_1_margin_$i --logpt  ./save_random_200/logs/cifar10_distill_lamda_0.1_local_lamda_1_margin_$i
#
#  python emailUtil.py --name cifar10_distill_lamda_0.1_local_lamda_1_margin_$i
#done




  for margin in -0.05 -0.1 -0.15 -0.2;
    do
    exp_name=${margin}_-0.05${distill_lamda}
    echo ${exp_name}
    python main_drift_etf.py --exp_name ${exp_name} --margin ${margin} --drift_lamda 0.05 --lamda 1 --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
    python main_linear_buffer.py --dataset cifar10 --learning_rate 0.5 --target_task 4 --ckpt  ./save_random_200/cifar10_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}
    python emailUtil.py --name ${exp_name}
  done