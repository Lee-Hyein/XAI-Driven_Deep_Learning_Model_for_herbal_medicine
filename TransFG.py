import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_attn=False):
        """Return (out, attn) when return_attn=True; otherwise return only out."""
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # => (B, N, 3, h, d)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # => (3, B, h, N, d)

        # scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attn_weights = self.attn_drop(attn)
        out = attn_weights @ v  # (B, h, N, d)

        # Merge the attention heads back together
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)

        if return_attn:
            return out, attn_weights  # (B, N, C), (B, h, N, N)
        return out


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attn_drop, qkv_bias)
        self.drop_path = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLPBlock(embed_dim, hidden_dim, drop)

    def forward(self, x, return_attn=False):
        """Return (x, attn) when return_attn=True; otherwise return only x."""
        x_norm = self.norm1(x)
        if return_attn:
            attn_out, attn = self.attn(x_norm, return_attn=True)
        else:
            attn_out = self.attn(x_norm)
            attn = None

        x = x + self.drop_path(attn_out)
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = x + self.drop_path(mlp_out)

        if return_attn:
            return x, attn
        return x


################################################
# 2. TransFG Model (Global + Local Branch)
################################################
class TransFGModel(nn.Module):
    """TransFG model with global and local branches."""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=3,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        top_k=8,
        local_tokens=4,
        drop_rate=0.1,
        attn_drop_rate=0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.top_k = top_k
        self.local_tokens = local_tokens

        # ----------------- Patch Embedding ----------------- #
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # ----------------- Position Embedding ----------------- #
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # local tokens
        self.local_tokens_param = nn.Parameter(torch.zeros(1, local_tokens, embed_dim))

        # ----------------- Transformer Blocks ----------------- #
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio,
                             drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ----------------- Heads ----------------- #
        self.global_head = nn.Linear(embed_dim, num_classes)

        # Local branch blocks (using half of the depth as an example)
        self.local_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio,
                             drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth // 2)
        ])
        self.local_norm = nn.LayerNorm(embed_dim)
        self.local_head = nn.Linear(embed_dim, num_classes)

        # Fusion
        self.fuse_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Patch importance computation (attention-based approximation instead of gradients)
        self.patch_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.local_tokens_param, std=0.02)
        
        # Initialize all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_pretrained_weights(self, pretrained_path=None):
        """Load pretrained ViT weights"""
        if pretrained_path is None:
            # Example of using timm to load pretrained weights
            try:
                import timm
            except ImportError:
                print("timm library not found. Installing timm...")
                import subprocess
                subprocess.check_call(["pip", "install", "timm"])
                import timm
                print("timm library installed successfully.")
                
            pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            
            # Copy patch-embedding weights
            self.patch_embed.weight.data.copy_(pretrained_vit.patch_embed.proj.weight.data)
            self.patch_embed.bias.data.copy_(pretrained_vit.patch_embed.proj.bias.data)
            
            # Copy positional-embedding weights
            self.pos_embed.data.copy_(pretrained_vit.pos_embed.data)
            
            # Copy transformer block weights
            for i, blk in enumerate(self.blocks):
                if i < len(pretrained_vit.blocks):
                    # Attention weights
                    blk.attn.qkv.weight.data.copy_(pretrained_vit.blocks[i].attn.qkv.weight.data)
                    blk.attn.qkv.bias.data.copy_(pretrained_vit.blocks[i].attn.qkv.bias.data)
                    blk.attn.proj.weight.data.copy_(pretrained_vit.blocks[i].attn.proj.weight.data)
                    blk.attn.proj.bias.data.copy_(pretrained_vit.blocks[i].attn.proj.bias.data)
                    
                    # MLP weights
                    blk.mlp.fc1.weight.data.copy_(pretrained_vit.blocks[i].mlp.fc1.weight.data)
                    blk.mlp.fc1.bias.data.copy_(pretrained_vit.blocks[i].mlp.fc1.bias.data)
                    blk.mlp.fc2.weight.data.copy_(pretrained_vit.blocks[i].mlp.fc2.weight.data)
                    blk.mlp.fc2.bias.data.copy_(pretrained_vit.blocks[i].mlp.fc2.bias.data)
                    
                    # Layer-norm weights
                    blk.norm1.weight.data.copy_(pretrained_vit.blocks[i].norm1.weight.data)
                    blk.norm1.bias.data.copy_(pretrained_vit.blocks[i].norm1.bias.data)
                    blk.norm2.weight.data.copy_(pretrained_vit.blocks[i].norm2.weight.data)
                    blk.norm2.bias.data.copy_(pretrained_vit.blocks[i].norm2.bias.data)
            
            print("Pretrained ViT weights loaded successfully.")
        else:
            # Load weights from a saved checkpoint
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Weights loaded from checkpoint: {pretrained_path}")


    def compute_patch_importance(self, patch_embs):
        """Compute patch importance via attention scores"""
        # Compute attention scores over patch embeddings
        attn_scores = self.patch_attention(patch_embs)  # (B, N, 1)
        return attn_scores.squeeze(-1)  # (B, N)

    def forward(self, x, return_last_attn=False):
        """Return the last attention map as well when return_last_attn=True."""
        B = x.shape[0]
        
        # 1. Patch Embedding
        x_patch = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x_patch = x_patch.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        # 2. Global Branch
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_cat = torch.cat([cls_tokens, x_patch], dim=1)
        x_cat = x_cat + self.pos_embed[:, : x_cat.size(1), :]
        x_cat = self.pos_drop(x_cat)
        
        attn_to_return = None

        # Transformer blocks
        for i, blk in enumerate(self.blocks):
            # Extract attention from the final block
            if return_last_attn and (i == len(self.blocks) - 1):
                x_cat, attn_block = blk(x_cat, return_attn=True)
                attn_to_return = attn_block
            else:
                x_cat = blk(x_cat)
            
        x_cat = self.norm(x_cat)
        cls_embed = x_cat[:, 0]  # (B, embed_dim)
        patch_embs = x_cat[:, 1:]  # (B, N, embed_dim)
        
        global_logits = self.global_head(cls_embed)
        
        # 3. Patch importance (attention scores instead of gradients)
        patch_importance = self.compute_patch_importance(patch_embs)
        patch_topk_idx = torch.topk(patch_importance, self.top_k, dim=1)[1]
        
        # 4. Local Branch
        B, N, C = patch_embs.shape
        
        # Gather top-K patches
        idx_expanded = patch_topk_idx.unsqueeze(-1).expand(-1, -1, C)
        selected_patches = torch.gather(patch_embs, 1, idx_expanded)
        
        # local tokens
        local_tokens = self.local_tokens_param.expand(B, self.local_tokens, C)
        
        x_local = torch.cat([local_tokens, selected_patches], dim=1)
        
        for blk in self.local_blocks:
            x_local = blk(x_local)
        x_local = self.local_norm(x_local)
        
        # Average only the local tokens
        local_part = x_local[:, : self.local_tokens, :]
        local_feat = local_part.mean(dim=1)
        
        local_logits = self.local_head(local_feat)
        
        # 5. Fusion
        fused = torch.cat([cls_embed, local_feat], dim=1)
        fused_logits = self.fuse_mlp(fused)
        
        if return_last_attn:
            return global_logits, local_logits, fused_logits, attn_to_return
        return global_logits, local_logits, fused_logits

################################################
# 3. Training loop
################################################
import csv
import os
import time
import numpy as np
from datetime import datetime

def train_transfg_single_experiment(model, train_loader, val_loader,
                                   device="cuda", lr=1e-4, epochs=30,
                                   save_dir="model/transfg_checkpoints/251024"):
    # Optimizer and scheduler setup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    best_acc = 0.0
    
    # Containers for per-epoch metrics collected in this experiment
    epoch_list = []
    train_global_accs = []
    train_local_accs = []
    train_fused_accs = []
    train_losses = []
    val_global_accs = []
    val_local_accs = []
    val_fused_accs = []
    val_losses = []
    epoch_times = []
    total_times = []
    
    # Start measuring total training time
    total_start_time = time.time()

    for epoch in range(epochs):
        # Measure epoch start time
        epoch_start_time = time.time()
        
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct_global = 0
        correct_local = 0
        correct_fused = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Use the unified forward outputs
            global_logits, local_logits, fused_logits = model(images)
            
            # Compute losses
            global_loss = F.cross_entropy(global_logits, labels)
            local_loss = F.cross_entropy(local_logits, labels)
            fused_loss = F.cross_entropy(fused_logits, labels)
            
            # Weighted loss function
            total_loss_batch = 0.5 * global_loss + 0.2 * local_loss + 0.3 * fused_loss
            
            # Single backward pass
            total_loss_batch.backward()
            optimizer.step()
            
            # Compute accuracy
            _, pred_global = global_logits.max(dim=1)
            _, pred_local = local_logits.max(dim=1)
            _, pred_fused = fused_logits.max(dim=1)
            
            correct_global += pred_global.eq(labels).sum().item()
            correct_local += pred_local.eq(labels).sum().item()
            correct_fused += pred_fused.eq(labels).sum().item()
            
            bs = images.size(0)
            total_loss += total_loss_batch.item() * bs
            total_samples += bs
            
            # Update the progress bar
            pbar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'global_acc': f"{100.0 * correct_global/total_samples:.2f}%",
                'local_acc': f"{100.0 * correct_local/total_samples:.2f}%",
                'fused_acc': f"{100.0 * correct_fused/total_samples:.2f}%"
            })

        # Step the scheduler after each epoch
        scheduler.step()
        
        # Compute epoch duration
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        
        # Print epoch summary
        train_loss_epoch = total_loss / total_samples
        acc_global = 100.0 * correct_global / total_samples
        acc_local = 100.0 * correct_local / total_samples
        acc_fused = 100.0 * correct_fused / total_samples

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Loss: {train_loss_epoch:.4f}, "
              f"Global Acc: {acc_global:.2f}%, "
              f"Local Acc: {acc_local:.2f}%, "
              f"Fused Acc: {acc_fused:.2f}%, "
              f"Epoch Time: {epoch_time:.2f}s, "
              f"Total Time: {total_time:.2f}s")

        # Validation
        val_loss = 0.0
        val_acc_global = 0.0
        val_acc_local = 0.0
        val_acc_fused = 0.0
        
        if val_loader is not None:
            val_results = validate_transfg(model, val_loader, device=device, return_metrics=True)
            val_loss = val_results['loss']
            val_acc_global = val_results['global_acc']
            val_acc_local = val_results['local_acc']
            val_acc_fused = val_results['fused_acc']

            if val_acc_fused > best_acc:
                best_acc = val_acc_fused
                save_checkpoint(model, optimizer, epoch, best_acc,
                              os.path.join(save_dir, 'best_model.pth'))
        
        # Persist metrics
        epoch_list.append(epoch + 1)
        train_global_accs.append(acc_global)
        train_local_accs.append(acc_local)
        train_fused_accs.append(acc_fused)
        train_losses.append(train_loss_epoch)
        val_global_accs.append(val_acc_global)
        val_local_accs.append(val_acc_local)
        val_fused_accs.append(val_acc_fused)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)
        total_times.append(total_time)
            
        # Save checkpoints periodically
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_acc_fused if val_loader else acc_fused,
                          os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
    # Save checkpoint after the final epoch
    save_checkpoint(model, optimizer, epochs-1, val_acc_fused if val_loader else acc_fused,
                   os.path.join(save_dir, 'last_model.pth'))
    
    # Report total training time
    total_training_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_training_time:.2f}초 ({total_training_time/60:.2f}분)")
    print(f"Average epoch time: {total_training_time/epochs:.2f}초")
    
    # Return experiment metrics
    return {
        'epochs': epoch_list,
        'train_global_accs': train_global_accs,
        'train_local_accs': train_local_accs,
        'train_fused_accs': train_fused_accs,
        'train_losses': train_losses,
        'val_global_accs': val_global_accs,
        'val_local_accs': val_local_accs,
        'val_fused_accs': val_fused_accs,
        'val_losses': val_losses,
        'epoch_times': epoch_times,
        'total_times': total_times
    }

def calculate_std_deviation_transfg(experiments_data):
    """Compute standard deviation for TransFG experiment metrics"""
    num_experiments = len(experiments_data)
    num_epochs = len(experiments_data[0]['train_global_accs'])
    
    # Compute standard deviation for each epoch
    train_global_std = []
    val_global_std = []
    train_local_std = []
    val_local_std = []
    train_fused_std = []
    val_fused_std = []
    train_loss_std = []
    val_loss_std = []
    
    for epoch_idx in range(num_epochs):
        # Gather results for the same epoch across experiments
        epoch_train_global = [exp['train_global_accs'][epoch_idx] for exp in experiments_data]
        epoch_val_global = [exp['val_global_accs'][epoch_idx] for exp in experiments_data]
        epoch_train_local = [exp['train_local_accs'][epoch_idx] for exp in experiments_data]
        epoch_val_local = [exp['val_local_accs'][epoch_idx] for exp in experiments_data]
        epoch_train_fused = [exp['train_fused_accs'][epoch_idx] for exp in experiments_data]
        epoch_val_fused = [exp['val_fused_accs'][epoch_idx] for exp in experiments_data]
        epoch_train_losses = [exp['train_losses'][epoch_idx] for exp in experiments_data]
        epoch_val_losses = [exp['val_losses'][epoch_idx] for exp in experiments_data]
        
        # Calculate the standard deviation
        train_global_std.append(np.std(epoch_train_global))
        val_global_std.append(np.std(epoch_val_global))
        train_local_std.append(np.std(epoch_train_local))
        val_local_std.append(np.std(epoch_val_local))
        train_fused_std.append(np.std(epoch_train_fused))
        val_fused_std.append(np.std(epoch_val_fused))
        train_loss_std.append(np.std(epoch_train_losses))
        val_loss_std.append(np.std(epoch_val_losses))
    
    return (train_global_std, val_global_std, train_local_std, val_local_std,
            train_fused_std, val_fused_std, train_loss_std, val_loss_std)

def train_transfg(model, train_loader, val_loader,
                  device="cuda", lr=1e-4, epochs=30,
                  save_dir="model/transfg_checkpoints/251024"):
    """Train TransFG via three repeated experiments"""
    print("=== Start of TransFG experiment repeated 3 times ===")
    
    # Run the three repeated experiments
    experiments_data = []
    for exp_id in range(3):
        print(f"\n=== Experiment {exp_id + 1}/3 started ===")
        
        # Use different random seeds per experiment
        torch.manual_seed(42 + exp_id)
        np.random.seed(42 + exp_id)
        
        # Instantiate a fresh model
        model_copy = TransFGModel(
            img_size=224,
            patch_size=16,
            num_classes=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            top_k=8,
            local_tokens=4,
            drop_rate=0.1
        )
        model_copy.load_pretrained_weights(pretrained_path=None)
        
        # Derive the checkpoint directory per experiment
        exp_save_dir = os.path.join(save_dir, f'experiment_{exp_id+1}')
        os.makedirs(exp_save_dir, exist_ok=True)
        
        experiment_result = train_transfg_single_experiment(
            model_copy, train_loader, val_loader, device, lr, epochs, exp_save_dir
        )
        experiments_data.append(experiment_result)
    
    # Calculate the standard deviation
    (train_global_std, val_global_std, train_local_std, val_local_std,
     train_fused_std, val_fused_std, train_loss_std, val_loss_std) = calculate_std_deviation_transfg(experiments_data)
    
    # Use the first experiment as the representative curve (mean proxy)
    epochs = experiments_data[0]['epochs']
    train_global_accs = experiments_data[0]['train_global_accs']
    train_local_accs = experiments_data[0]['train_local_accs']
    train_fused_accs = experiments_data[0]['train_fused_accs']
    train_losses = experiments_data[0]['train_losses']
    val_global_accs = experiments_data[0]['val_global_accs']
    val_local_accs = experiments_data[0]['val_local_accs']
    val_fused_accs = experiments_data[0]['val_fused_accs']
    val_losses = experiments_data[0]['val_losses']
    epoch_times = experiments_data[0]['epoch_times']
    total_times = experiments_data[0]['total_times']
    
    # Build a CSV that includes the actual standard deviation
    results_dir = os.path.join(save_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    log_file = os.path.join(results_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Epoch', 'Train_Loss', 'Train_Global_Acc', 'Train_Local_Acc', 'Train_Fused_Acc', 
                 'Val_Loss', 'Val_Global_Acc', 'Val_Local_Acc', 'Val_Fused_Acc', 'Epoch_Time', 'Total_Time',
                 'Train_Global_Acc_ErrorBar', 'Val_Global_Acc_ErrorBar', 'Train_Local_Acc_ErrorBar', 'Val_Local_Acc_ErrorBar',
                 'Train_Fused_Acc_ErrorBar', 'Val_Fused_Acc_ErrorBar', 'Train_Loss_ErrorBar', 'Val_Loss_ErrorBar']
        writer.writerow(header)
        
        for i in range(len(epochs)):
            row = [epochs[i], train_losses[i], train_global_accs[i], train_local_accs[i], train_fused_accs[i],
                  val_losses[i], val_global_accs[i], val_local_accs[i], val_fused_accs[i], 
                  epoch_times[i], total_times[i],
                  train_global_std[i], val_global_std[i], train_local_std[i], val_local_std[i],
                  train_fused_std[i], val_fused_std[i], train_loss_std[i], val_loss_std[i]]
            writer.writerow(row)
    
    # Generate the plots that include standard deviation
    create_training_plots_with_std(log_file, results_dir, 
                                  train_global_std, val_global_std, train_local_std, val_local_std,
                                  train_fused_std, val_fused_std, train_loss_std, val_loss_std)
    
    print(f"TransFG training completed with 3 repeated experiments.")
    print(f"Actual standard deviation calculated and used in Error Bar.")
    print(f"Results saved to: {log_file}")
    
    return log_file

def create_training_plots_with_std(log_file, save_dir, 
                                  train_global_std, val_global_std, train_local_std, val_local_std,
                                  train_fused_std, val_fused_std, train_loss_std, val_loss_std):
    """Visualize training metrics while using actual standard deviation"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Configure matplotlib fonts
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Read metrics from CSV
    df = pd.read_csv(log_file)
    
    # Accuracy plots (real standard deviation)
    plt.figure(figsize=(15, 10))
    
    # Global accuracy
    plt.subplot(2, 2, 1)
    plt.errorbar(df['Epoch'], df['Train_Global_Acc'], yerr=train_global_std, 
                fmt='b-o', label='Training Global Accuracy', capsize=3, capthick=1)
    plt.errorbar(df['Epoch'], df['Val_Global_Acc'], yerr=val_global_std, 
                fmt='r-o', label='Validation Global Accuracy', capsize=3, capthick=1)
    plt.title('Global Accuracy (Actual Standard Deviation Error Bar)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Local accuracy
    plt.subplot(2, 2, 2)
    plt.errorbar(df['Epoch'], df['Train_Local_Acc'], yerr=train_local_std, 
                fmt='g-o', label='Training Local Accuracy', capsize=3, capthick=1)
    plt.errorbar(df['Epoch'], df['Val_Local_Acc'], yerr=val_local_std, 
                fmt='orange', marker='o', label='Validation Local Accuracy', capsize=3, capthick=1)
    plt.title('Local Accuracy (Actual Standard Deviation Error Bar)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Fused accuracy
    plt.subplot(2, 2, 3)
    plt.errorbar(df['Epoch'], df['Train_Fused_Acc'], yerr=train_fused_std, 
                fmt='purple', marker='o', label='Training Fused Accuracy', capsize=3, capthick=1)
    plt.errorbar(df['Epoch'], df['Val_Fused_Acc'], yerr=val_fused_std, 
                fmt='brown', marker='o', label='Validation Fused Accuracy', capsize=3, capthick=1)
    plt.title('Fused Accuracy (Actual Standard Deviation Error Bar)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Loss plots
    plt.subplot(2, 2, 4)
    plt.errorbar(df['Epoch'], df['Train_Loss'], yerr=train_loss_std, 
                fmt='b-o', label='Training Loss', capsize=3, capthick=1)
    plt.errorbar(df['Epoch'], df['Val_Loss'], yerr=val_loss_std, 
                fmt='r-o', label='Validation Loss', capsize=3, capthick=1)
    plt.title('Loss (Actual Standard Deviation Error Bar)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, 'transfg_training_results.png'), dpi=300)
    plt.show()
    
    print(f"TransFG training results saved to: {os.path.join(save_dir, 'transfg_training_results.png')}")
    
def save_checkpoint(model, optimizer, epoch, acc, filepath):
    """Save model checkpoints"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath, device="cuda"):
    """Load model checkpoints"""
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return None
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"with accuracy {checkpoint['accuracy']:.2f}%")
    return checkpoint

def validate_transfg(model, val_loader, device="cuda", return_metrics=False):
    """Validation helper with optional metric return"""
    model.eval()
    total_samples = 0
    correct_global = 0
    correct_local = 0
    correct_fused = 0
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Use the unified forward outputs
            global_logits, local_logits, fused_logits = model(images)
            
            # Compute losses
            global_loss = F.cross_entropy(global_logits, labels)
            local_loss = F.cross_entropy(local_logits, labels)
            fused_loss = F.cross_entropy(fused_logits, labels)
            batch_loss = 0.5 * global_loss + 0.2 * local_loss + 0.3 * fused_loss
            total_loss += batch_loss.item() * labels.size(0)
            
            # Compute accuracy
            _, pred_global = global_logits.max(dim=1)
            _, pred_local = local_logits.max(dim=1)
            _, pred_fused = fused_logits.max(dim=1)
            
            correct_global += pred_global.eq(labels).sum().item()
            correct_local += pred_local.eq(labels).sum().item()
            correct_fused += pred_fused.eq(labels).sum().item()
            
            total_samples += labels.size(0)
            
            # Update the progress bar
            pbar.set_postfix({
                'global_acc': f"{100.0 * correct_global/total_samples:.2f}%",
                'local_acc': f"{100.0 * correct_local/total_samples:.2f}%",
                'fused_acc': f"{100.0 * correct_fused/total_samples:.2f}%"
            })

    # Compute validation metrics
    val_loss = total_loss / total_samples
    acc_global = 100.0 * correct_global / total_samples
    acc_local = 100.0 * correct_local / total_samples
    acc_fused = 100.0 * correct_fused / total_samples
    
    print(f"Validation: "
          f"Loss = {val_loss:.4f}, "
          f"Global Acc = {acc_global:.2f}%, "
          f"Local Acc = {acc_local:.2f}%, "
          f"Fused Acc = {acc_fused:.2f}%")

    if return_metrics:
        return {
            'loss': val_loss,
            'global_acc': acc_global,
            'local_acc': acc_local,
            'fused_acc': acc_fused
        }
    return acc_fused

################################################
# 5. main() - argparse + dataloader
################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, 
                       default="/mnt/mydisk/hyein/TransFG_data/dataset_cls_PE_aug/train",
                       help='train data directory')
    parser.add_argument("--val_dir", type=str,
                       default="/mnt/mydisk/hyein/TransFG_data/dataset_cls_PE_aug/val",
                       help='validation data directory (optional)')
    parser.add_argument("--num_classes", type=int, default=3,
                       help='number of classes')
    parser.add_argument("--batch_size", type=int, default=16,
                       help='batch size')
    parser.add_argument("--epochs", type=int, default=30,
                       help='number of epochs')

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/251024")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--pretrained", type=str, default=None,
                       help="Path to pretrained model (if not provided, uses timm library)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training data: {args.train_dir}")
    print(f"Validation data: {args.val_dir}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.epochs}")

    # Apply stronger data augmentation
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Dataset and dataloader configuration
    train_dataset = datasets.ImageFolder(args.train_dir, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)

    val_loader = None
    if args.val_dir is not None:
        val_dataset = datasets.ImageFolder(args.val_dir, transform=transform_val)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)

    # Adjust model capacity
    model = TransFGModel(
        img_size=224,
        patch_size=16,
        num_classes=args.num_classes,
        embed_dim=768,  # Align with the pretrained ViT configuration
        depth=12,       # Align with the pretrained ViT configuration
        num_heads=12,   # Align with the pretrained ViT configuration
        mlp_ratio=4.0,
        top_k=8,
        local_tokens=4,
        drop_rate=0.1
    )

    model.load_pretrained_weights(pretrained_path=args.pretrained)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Load checkpoint if requested
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, args.resume)
        if checkpoint:
            print(f"Resuming from checkpoint: {args.resume}")
    
    # Training entry point
    train_transfg(model, train_loader, val_loader,
                 device=device,
                 lr=5e-5,  # Reduced learning rate
                 epochs=args.epochs,  # Use the epoch count provided via CLI
                 save_dir=args.checkpoint_dir)
    

if __name__ == "__main__":
    main()
