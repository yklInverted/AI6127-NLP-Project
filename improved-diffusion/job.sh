#!/bin/sh

#SBATCH -n 1 # 指定核心数量
#SBATCH -N 1 # 指定node的数量
#SBATCH -t 0-5:00 # 运行总时间，天数-小时数-分钟， D-HH:MM
#SBATCH -p SCSEGPU_M1 # 提交到哪一个分区
#SBATCH --mem=2000 # 所有核心可以使用的内存池大小，MB为单位
#SBATCH -o myjob.o # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e myjob.e # 把报错结果STDERR保存在哪一个文件
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=S220048@e.ntu.edu.sg # 把通知发送到哪一个邮箱
#SBATCH --gres=gpu:1 # 需要使用多少GPU，n是需要的数量
#SBATCH --job-name=Diffusion!!


nvidia-smi
python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 200000  --seed 102 --noise_schedule sqrt --in_channel 16 --modality e2e-tgt --submit no --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 821  --e2e_train ../datasets/e2e_data " --notes xstart_e2e