gpu=0

for seed in 1 2 3 4 5
do

python regression.py \
    --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
    --dataset activity --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_spd_forecast --gpu $gpu --dropout 0.1 \
    --history 3000 --task forecasting --collate indseq \
    --dist_alpha 1.0 --dist_beta 1.0 --spd_dim 4

done
