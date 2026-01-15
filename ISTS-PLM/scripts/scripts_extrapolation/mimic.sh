
gpu=0

for seed in 1 2 3 4 5
do
python regression.py \
    --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
    --dataset mimic --seed $seed --d_model 768 --max_len -1 \
    --model istsplm_forecast --n_te_plmlayer 3 \
    --gpu $gpu --n_st_plmlayer 3  --dropout 0.1 \
    --te_model bert --st_model bert --sample_rate 1 --semi_freeze\
    --history 24 --task forecasting

# python regression.py \
#     --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
#     --dataset mimic --seed $seed --d_model 768 --max_len -1 \
#     --model istsplm_vector_forecast --collate vector --n_te_plmlayer 3 \
#     --gpu $gpu --n_st_plmlayer 3  --dropout 0.1 \
#     --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
#     --history 24 --task forecasting

# python regression.py \
#     --batch 6 --lr 5e-4 --state 'def' --epoch 1000 --patience 10 \
#     --dataset mimic --seed $seed --d_model 768 --max_len -1 \
#     --model istsplm_set_forecast --collate vector --n_te_plmlayer 3 \
#     --gpu $gpu --n_st_plmlayer 3  --dropout 0.1 \
#     --te_model gpt --st_model bert --sample_rate 1 --semi_freeze\
#     --history 24 --task forecasting
done

# python regression.py --batch 6 --lr 5e-5 --state "def" --epoch 1000 --patience 20 --dataset mimic --seed 42 --d_model 128 --max_len -1 --model istsplm_spd_forecast --gpu 0 --dropout 0.1 --history 24 --task forecasting --collate indseq --dist_alpha 0.5 --dist_beta 1.0 --spd_dim 2