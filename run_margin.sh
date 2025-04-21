for local_lamda in 1 8;
 do
 for distill_lamda in 0.5 0.1 0.05;
   do
   exp_name=cifar100_distill_lamda_${distill_lamda}_local_lamda_${local_lamda}_500

   python main_drift_etf.py --exp_name ${exp_name} --drift_lamda ${distill_lamda} --lamda ${local_lamda} --batch_size 512 --model resnet18 --dataset cifar100-imagenet --mem_size 200 --epochs 50  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;

   python main_linear_buffer.py --dataset cifar100 --learning_rate 0.1 --target_task 9 --ckpt  ./save_random_200/cifar100_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}

   python emailUtil.py --name ${exp_name}

 done
done


for local_lamda in 0.1 0.5;
 do
 for distill_lamda in 0.5 0.1 0.05;
   do
   exp_name=cifar100_distill_lamda_${distill_lamda}_local_lamda_${local_lamda}_500

   python main_drift_etf.py --exp_name ${exp_name} --drift_lamda ${distill_lamda} --lamda ${local_lamda} --batch_size 512 --model resnet18 --dataset cifar100-imagenet --mem_size 200 --epochs 50  --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;

   python main_linear_buffer.py --dataset cifar100 --learning_rate 0.1 --target_task 9 --ckpt  ./save_random_200/cifar100_models/${exp_name} --logpt  ./save_random_200/logs/${exp_name}

   python emailUtil.py --name ${exp_name}

 done
done