model_name='DPC_L2B'
noise_type='sym'
gpuid='0'
seed='123'
save_path='./logs/'
data_path='../data/nyb/DPC'
config_path='./configs/DPC_L2B_CIFAR.py'
dataset='cifar-10'
num_classes=10
warmup_epoch=10
need_clean=True
single_meta=1
noise_rate_lambda_u_pairs=(0.8 25)  # 第一对噪声率和lambda_u
#noise_rate_lambda_u_pairs=(0.5 25)  # 第二对噪声率和lambda_u
#noise_rate_lambda_u_pairs+=(0.8 25) # 第三对

for i in "${!noise_rate_lambda_u_pairs[@]}"; do
    if (( i % 2 == 0 )); then
        noise_rate=${noise_rate_lambda_u_pairs[$i]}
        lambda_u=${noise_rate_lambda_u_pairs[$i+1]}
        python main.py -c=$config_path --save_path=$save_path --noise_type=$noise_type --seed=$seed --gpu=$gpuid --percent=$noise_rate --dataset=$dataset --num_classes=$num_classes --root=$data_path --model_name=$model_name --lambda_u=$lambda_u --warmup_epoch=$warmup_epoch --need_clean=$need_clean --single_meta=$single_meta
    fi
done
