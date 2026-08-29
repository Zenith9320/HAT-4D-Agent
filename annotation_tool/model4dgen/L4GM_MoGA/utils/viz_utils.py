# Modified by HAT-4D; see the subtree MODIFICATIONS.md.
import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from typing import Optional, Tuple

def save_images_clip_as_video(images_clip, CN, V, save_path="images_clip.mp4", fps=5):
    B, _, C, H, W = images_clip.shape
    images_clip = images_clip[0].detach().cpu()

    images_clip = images_clip.reshape(CN, V, C, H, W)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    images_clip = images_clip.permute(0,3,1,4,2).reshape(CN,H, W*V,C)
    imageio.mimwrite(save_path, images_clip, fps=2)


    print(f"Saved {save_path} ({CN} frames, {V} views, {H}x{W})")
    
    
# ====== Group Viz ========
def logits_to_groups_with_capacity(
    router_logits: torch.Tensor, 
    capacity_factor: float = 1.0, 
    T: float = 1.0
) -> torch.Tensor:
    
    B_prime, L, M = router_logits.shape
    N = B_prime * L
    
    weights = (router_logits / T).softmax(dim=-1)        # [B', L, M]
    _, groups = torch.topk(weights, k=1, dim=-1)
    
    groups_flat  = groups.reshape(-1)                    # [N]

    # 2. Hard Capacity
    capacity = max(1, int(capacity_factor * (N / M)))

    counts = torch.zeros(M, dtype=torch.int32, device=groups_flat.device)

    final_group = torch.empty_like(groups_flat)

    for idx in range(N):
        g = groups_flat[idx].item()
        
        if counts[g] < capacity:
            final_group[idx] = g
            counts[g] += 1
        else:
            alt = torch.argmin(counts).item() 
            final_group[idx] = alt
            counts[alt] += 1
            
    return final_group


def get_group_map_from_flat(num_clip_frames,num_views, final_group, B, H, W):
    
    # [B, L] -> [B, T, V, H, W]
    group_map_reshaped = final_group.reshape(B,num_clip_frames,num_views, H, W)
    
    # [B, T, V, H, W] -> [BTV, H, W]
    BTV = B * num_clip_frames * num_views
    print("group_map:", group_map_reshaped.shape)
    print("BTV, H, W:", (BTV, H, W))
    group_map = group_map_reshaped.reshape(BTV, H, W) 
    return group_map


def visualize_moga_groups_multiview(
    group_map: torch.Tensor,
    original_input_shape: Tuple[int, int, int, int], # [BTV, C, H_orig, W_orig]
    num_frames: int,
    num_views: int,
    sample_index: int = 0,
    save_path: Optional[str] = None
):
    if group_map is None:
        print("Error: group_map is None.")
        return

    BTV, C_orig, H_orig, W_orig = original_input_shape
    
    B_orig = BTV // (num_frames * num_views)
    if sample_index >= B_orig:
        raise IndexError(f"Sample index {sample_index} out of bounds for original batch size {B_orig}")

    start_idx = sample_index * (num_frames * num_views)
    end_idx = start_idx + (num_frames * num_views)
    
    target_groups_flat = group_map[start_idx:end_idx]
    H_sub, W_sub = target_groups_flat.shape[1:]
    
    groups_tv = target_groups_flat.reshape(num_frames, num_views, H_sub, W_sub).cpu().numpy()
    
    num_groups = int(group_map.max().item()) + 1
    cmap = plt.cm.get_cmap('Spectral', num_groups) 
    
    fig, axes = plt.subplots(
        nrows=num_frames, 
        ncols=num_views, 
        figsize=(4 * num_views, 4 * num_frames), 
        squeeze=False
    )

    sample_rate_h = H_orig / H_sub
    sample_rate_w = W_orig / W_sub
    
    for t in range(num_frames):
        for v in range(num_views):
            ax = axes[t, v]
            group_ids = groups_tv[t, v]
            
            im = ax.imshow(group_ids, cmap=cmap, interpolation='nearest')
            
            ax.set_title(f"Frame {t}, View {v}", fontsize=10)
            ax.axis('off')

    fig.suptitle(
        f"MoGA Token Grouping Map (Sample {sample_index}) | Feature Size: {H_sub}x{W_sub}, Groups: {num_groups}", 
        fontsize=14
    )
    
    bounds = np.linspace(-0.5, num_groups - 0.5, num_groups + 1)
    ticks = np.arange(num_groups)
    
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(group_map.cpu().numpy().flatten())
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_yticklabels([str(int(t)) for t in ticks], fontsize=8)
    cbar.set_label('Token Group ID (M)', rotation=270, labelpad=15, fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    if save_path:
        plt.savefig(save_path)
        print(f"Group map saved to {save_path}")
    else:
        plt.show()
