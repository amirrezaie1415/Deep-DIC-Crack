# MIT License
# Copyright (c) 2021 
# Earthquake Engineering and Structural Dynamics (EESD), EPFL 

declare -a model=("TernausNet16")

for model_name in "${model[@]}"
do
for weight_decay in 0
do
for learning_rate in  2e-4 # put 9e-5 for pretrained=0 and 2e-4 for pretrained=1
do

python run.py --model_type="$model_name" --lr=$learning_rate --weight_decay=$weight_decay --num_epochs=100 --pretrained=1  --batch_size=1

done
done
done



