

#0.1 0.5 1
#for i in 0.1 0.5 1 2 3 4 5;
#  do echo $i;
#  python main_drift_etf.py --exp_name cifar10_distill_lamda_0.1_local_lamda_$i --drift_lamda 0.1  --lamda $i --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
#  python main_linear_buffer.py --dataset cifar10 --learning_rate 0.1 --target_task 4 --ckpt  ./save_random_200/cifar10_models/cifar10_distill_lamda_0.1_local_lamda_$i --logpt  ./save_random_200/logs/cifar10_distill_lamda_0.1_local_lamda_$i
#
#  python emailUtil.py --name cifar10_distill_lamda_0.1_local_lamda_$i
#done


#

for local_lamda in 0.1 0.5 1 2;
  do
  for distill_lamda in 0.5 0.1 0.05;
    do
    echo cifar10_distill_lamda_${distill_lamda}_local_lamda_${local_lamda}
    python main_drift_etf.py --exp_name cifar10_distill_lamda_${distill_lamda}_local_lamda_${local_lamda} --drift_lamda ${distill_lamda} --lamda ${local_lamda} --batch_size 512 --model resnet18 --dataset tiny-imagenet --mem_size 200 --epochs 50  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
    python main_linear_buffer.py --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt  ./save_random_200/tiny-imagenet_models/cifar10_distill_lamda_${distill_lamda}_local_lamda_${local_lamda} --logpt  ./save_random_200/logs/cifar10_distill_lamda_${distill_lamda}_local_lamda_${local_lamda}
    python emailUtil.py --name cifar10_distill_lamda_${distill_lamda}_local_lamda_${local_lamda}
  done
done