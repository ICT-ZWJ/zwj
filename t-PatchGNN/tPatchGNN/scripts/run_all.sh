patience=10
gpu=0

# for seed in 1
# do
#     python run_models.py \
#     --dataset ushcn --state 'def' --history 24 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 24 --stride 24 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 64 \
#     --outlayer Linear --seed $seed --gpu $gpu \
#     --model ISTS_tPatch
# done

for seed in 1
do
    python run_models.py \
    --dataset physionet --state 'def' --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu \
    --model ISTS_tPatch
  done

# for seed in 1
# do
#     python run_models.py \
#     --dataset mimic --state 'def' --history 24 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 64 \
#     --outlayer Linear --seed $seed --gpu $gpu
#     --model ISTS_tPatch
#   done

# for seed in 1
# do
#     python run_models.py \
#     --dataset activity --state 'def' --history 3000 \
#     --patience $patience --batch_size 32 --lr 1e-3 \
#     --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
#     --te_dim 10 --node_dim 10 --hid_dim 32 \
#     --outlayer Linear --seed $seed --gpu $gpu \
#     --model ISTS_tPatch
#     done

# patience=10
# gpu=0

# for seed in 1
# do
#   for dataset in ushcn physionet mimic activity
#   do
#     case "$dataset" in
#       ushcn)
#         history=24
#         lr=1e-3
#         patch_size=24
#         stride=24
#         hid_dim=64
#         ;;
#       physionet)
#         history=24
#         lr=1e-4
#         patch_size=8
#         stride=8
#         hid_dim=64
#         ;;
#       mimic)
#         history=24
#         lr=1e-3
#         patch_size=8
#         stride=8
#         hid_dim=64
#         ;;
#       activity)
#         history=3000
#         lr=1e-3
#         patch_size=300
#         stride=300
#         hid_dim=32
#         ;;
#       *)
#         echo "Unknown dataset: $dataset"
#         exit 1
#         ;;
#     esac

#     python run_models.py \
#       --dataset "$dataset" --state 'def' --history "$history" \
#       --patience "$patience" --batch_size 32 --lr "$lr" \
#       --patch_size "$patch_size" --stride "$stride" --nhead 1 --tf_layer 1 --nlayer 1 \
#       --te_dim 10 --node_dim 10 --hid_dim "$hid_dim" \
#       --outlayer Linear --seed "$seed" --gpu "$gpu" \
#       --model ISTS_tPatch
#   done
# done
