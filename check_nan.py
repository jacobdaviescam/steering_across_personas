import torch
d = torch.load('outputs/gemma-2-27b-it/activations/con_artist_assertiveness_pos.pt',
map_location='cpu', weights_only=True)
total = 0; nans = 0
for k, v in d.items():
    total += 1
    if v.isnan().any():
        nans += 1
print(f'{nans}/{total} tensors contain NaN')
