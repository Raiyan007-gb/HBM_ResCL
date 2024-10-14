#!/bin/bash
#SBATCH --job-name=Inat
#SBATCH --mail-user=jqcui@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=Inat.log
#SBATCH --gres=gpu:4
#SBATCH -c 40 
#SBATCH --constraint=ubuntu18,highcpucount
#SBATCH -p batch_72h 
#SBATCH -w gpu47

# source activate py3.6pt1.5

python iNaturalTrain_reslt.py \
  --arch resnet50_reslt \
  --mark resnet50_reslt_bt256 \
  -dataset iNaturalist2018 \
  --data_path /media/intisar/dataset/visual_categorization/herbarium-2022-fgvc9/ \
  -b 112 \
  --epochs 210 \
  --num_works 20 \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --beta 0.85 \
  --gamma 0.3 \
  --after_1x1conv \
  --num_classes 15505 \
  --resume data/iNaturalist2018/resnet50_reslt_bt256/model_best.pth.tar 
