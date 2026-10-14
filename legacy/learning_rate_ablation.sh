#0.01 0.03 0.05 0.1 0.2 0.3
for i in 0.5 0.1 0.05;
  do echo $i;
  python3 main_drift_etf.py --exp_name tiny_lamda_0.05_1_lr_$i --drift_lamda 0.05  --lamda 1 --batch_size 512 --model resnet18 --dataset tiny-imagenet --mem_size 200 --epochs 100  --learning_rate $i --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine;
  python3 emailUtil.py --name cifar10_lamda_0.05_1_lr_$i
done



