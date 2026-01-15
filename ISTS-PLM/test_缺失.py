import torch

path = r"E:\ITS\ISTS-PLM\data\physionet\processed\set-a_0.0.pt"
data = torch.load(path, map_location="cpu")  # list of (record_id, tt, vals, mask)

total_obs = 0.0
total_cnt = 0
per_var_obs = None
per_var_cnt = None

for _, _, _, mask in data:
    mask = mask.float()  # [T, D]
    total_obs += mask.sum().item()
    total_cnt += mask.numel()

    if per_var_obs is None:
        per_var_obs = mask.sum(dim=0)
        per_var_cnt = torch.ones_like(mask).sum(dim=0)
    else:
        per_var_obs += mask.sum(dim=0)
        per_var_cnt += torch.ones_like(mask).sum(dim=0)

overall_missing = 1.0 - total_obs / total_cnt
per_var_missing = 1.0 - (per_var_obs / per_var_cnt)

print("Overall missing rate:", overall_missing)
print("Per-variable missing rate:", per_var_missing.tolist())
