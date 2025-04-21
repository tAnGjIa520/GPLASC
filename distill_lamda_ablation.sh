


#for i in 0.01 0.03 0.05 0.1 0.2 0.3 0.5;
#  do echo $i;
#  python main_drift_etf.py --exp_name cifar10_lamda_0.05_1_distill_lamda_$i --drift_lamda $i  --lamda 1 --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
#done


#for i in 0.01 0.03 0.05 0.1 0.2 0.3 0.5;
#  do echo $i;
#  python main_drift_etf.py --exp_name cifar10_distill_lamda_$i_local_lamda_1 --drift_lamda $i  --lamda 1 --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
#  python main_linear_buffer.py --dataset cifar10 --learning_rate 0.1 --target_task 4 --ckpt  ./save_random_200/cifar10_models/cifar10_distill_lamda_$i_local_lamda_1 --logpt  ./save_random_200/logs/cifar10_distill_lamda_$i_local_lamda_1
#  python emailUtil.py --name learning_rate_finished
#done

for i in 0.5 0.1 0.05;
  do echo $i;
  python main_drift_etf_tiny.py --exp_name tiny_distill_lamda_${i}_local_lamda_1 --drift_lamda $i  --lamda 1 --batch_size 512 --model resnet18 --dataset tiny-imagenet --mem_size 200 --epochs 50  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
  python main_linear_buffer.py --dataset tiny-imagenet --learning_rate 0.1 --target_task 9 --ckpt  ./save_random_200/tiny-imagenet_models/tiny_distill_lamda_$i_local_lamda_1 --logpt ./save_random_200/logs/tiny_distill_lamda_$i_local_lamda_1
  python emailUtil.py --name distill_$i_finish
done