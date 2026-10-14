#! /bin/bash
read_dir(){
#    ./save_weight_200_/cifar10_models/
    path=./save_random_$1/$2_models/

    for file in `ls $path`       #注意此处这是两个反引号，表示运行系统命令
    do

    # 直接引用变量

      if [ $2 == 'cifar10' ]
      then
        echo "python main_linear_buffer.py --dataset $2 --learning_rate 0.03 --target_task 4 --ckpt  ./save_random_$1/$2_models/$file --logpt  ./save_random_$1/logs/$file"
        python main_linear_buffer.py --dataset $2 --learning_rate 0.03 --target_task 4 --ckpt  ./save_random_$1/$2_models/$file --logpt  ./save_random_$1/logs/$file
        python emailUtil.py --name $file

      elif [ $2 == 'cifar100' ]
      then
        python main_linear_buffer.py --dataset $2 --learning_rate 0.03 --target_task 4 --ckpt  ./save_random_$1/$2_models/$file --logpt  ./save_random_$1/logs/$file
      elif [ $2 == 'tiny-imagenet' ]
      then
        python main_linear_buffer.py --dataset $2 --learning_rate 0.03 --target_task 4 --ckpt  ./save_random_$1/$2_models/$file --logpt  ./save_random_$1/logs/$file

      fi

    done
}
#读取第一个参数
read_dir $1 $2
# 2 exampler的数量 3数据集的名字
# linear_for_run.sh 200 cifar10

#python main_linear_buffer_cifar100.py --dataset cifar100 --learning_rate 0.03 --target_task 4 --ckpt  ./save_random_200/cifar100_models/cifar100_etf_20_5_0.1_1 --logpt  ./save_random_200/logs/cifar100_etf_20_5_0.1_1
#python main_linear_buffer.py --dataset cifar10 --learning_rate 0.1 --target_task 4 --ckpt  ./save_random_200/cifar10_models/cifar10_lamda_0.05_1_lr_0.03 --logpt  ./save_random_200/logs/cifar10_lamda_0.05_1_lr_0.03