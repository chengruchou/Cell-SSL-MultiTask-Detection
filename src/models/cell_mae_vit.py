# cell_mae_vit_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# 基礎元件：Patch Embedding + Transformer Block
# --------------------------------------------------

class PatchEmbed(nn.Module):
    """
    將影像切成 patch，做成 token。
    img_size: 輸入影像大小 (假設正方形)
    patch_size: patch 邊長
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        # Conv2d 做 patchify + 線性投影
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                 # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2) # (B, N, embed_dim)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads,
            dropout=attn_drop,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x):
        # x: (B, N, C)
        x_res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x_res + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------
# ViT Encoder（會被 MAE 和下游任務共用）
# --------------------------------------------------

class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, C)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat((cls, x), dim=1)          # (B, 1+N, C)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x  # (B, 1+N, C)


# --------------------------------------------------
# MAE：Self-Supervised Pretrain
# --------------------------------------------------

class MAE(nn.Module):
    """
    簡化版 MAE：
    - encoder: ViTEncoder
    - decoder: 較小的 transformer，重建被 mask 的 patch
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        decoder_dim=192,
        decoder_depth=4,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
        )
        self.mask_ratio = mask_ratio
        num_patches = self.encoder.patch_embed.num_patches
        self.num_patches = num_patches

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, num_heads=4)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(
            decoder_dim,
            patch_size * patch_size * in_chans,
            bias=True
        )

        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # ---- random masking ----
    def random_masking(self, x):
        """
        x: patch tokens (不含 cls) shape (B, N, C)
        回傳:
        - x_keep: 保留的 patch tokens
        - mask: (B, N) 1 表示被 mask
        - ids_restore: 可以還原原本順序的 index
        """
        B, N, C = x.shape
        len_keep = int(N * (1 - self.mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_keep = torch.gather(
            x, 1,
            ids_keep.unsqueeze(-1).expand(-1, -1, C)
        )

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_keep, mask, ids_restore

    # ---- MAE 前向 ----
    def forward(self, x):
        # encoder
        latent = self.encoder(x)                # (B, 1+N, C)
        cls_token, patch_tokens = latent[:, :1, :], latent[:, 1:, :]

        # masking
        x_keep, mask, ids_restore = self.random_masking(patch_tokens)

        # decoder input
        dec_tokens = self.decoder_embed(x_keep)  # (B, N_keep, C_dec)
        B, N_keep, C_dec = dec_tokens.shape

        # 補上 mask token
        mask_tokens = self.mask_token.expand(B, self.num_patches - N_keep, -1)
        dec_tokens_ = torch.cat([dec_tokens, mask_tokens], dim=1)  # (B, N, C_dec)

        # 還原原本 patch 的順序
        dec_tokens_ = torch.gather(
            dec_tokens_,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, C_dec),
        )

        # 加上 cls
        dec_tokens_ = torch.cat(
            [self.decoder_embed(cls_token), dec_tokens_],
            dim=1
        )
        dec_tokens_ = dec_tokens_ + self.decoder_pos_embed

        # decoder transformer
        for blk in self.decoder_blocks:
            dec_tokens_ = blk(dec_tokens_)
        dec_tokens_ = self.decoder_norm(dec_tokens_)

        # 預測 patch pixel
        dec_patches = self.decoder_pred(dec_tokens_[:, 1:, :])  # (B, N, P*P*C)
        return dec_patches, mask

    # ---- 重建 loss ----
    def loss(self, imgs, dec_patches, mask):
        """
        imgs: (B, C, H, W)
        dec_patches: (B, N, patch*patch*C)
        mask: (B, N) 1 表示該 patch 被 mask（只對這些算 loss）
        """
        B, C, H, W = imgs.shape
        patch_size = self.encoder.patch_embed.patch_size

        # 把原圖 patchify 成 (B, N, P*P*C)
        target = imgs.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        target = target.contiguous().view(B, C, -1, patch_size, patch_size)
        target = target.permute(0, 2, 1, 3, 4).contiguous()
        target = target.view(B, -1, C * patch_size * patch_size)

        # MSE on masked patches
        loss = ((dec_patches - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_loss(self, imgs):
        dec_patches, mask = self.forward(imgs)
        loss = self.loss(imgs, dec_patches, mask)
        return loss, dec_patches, mask


# --------------------------------------------------
# 共用 backbone：把 encoder 輸出整理成 CLS 向量 + feature map
# --------------------------------------------------

class CellViTBackbone(nn.Module):
    """
    包住 MAE 的 encoder，提供：
    - cls 向量：用於分類
    - feature map：用於 seg / det
    """
    def __init__(self, mae: MAE, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = mae.encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        tokens = self.encoder(x)      # (B, 1+N, C)
        cls = tokens[:, 0]            # (B, C)
        patch_tokens = tokens[:, 1:]  # (B, N, C)

        B, N, C = patch_tokens.shape
        grid_size = int(N ** 0.5)     # 假設正方形 patch grid

        feat_map = patch_tokens.transpose(1, 2).reshape(
            B, C, grid_size, grid_size
        )  # (B, C, h, w)
        return cls, feat_map


# --------------------------------------------------
# 下游任務 1：分類
# --------------------------------------------------

class CellClassifier(nn.Module):
    def __init__(self, backbone: CellViTBackbone, num_classes: int, use_cls_token: bool = True):
        super().__init__()
        self.backbone = backbone
        dim = self.backbone.encoder.norm.normalized_shape[0]
        self.use_cls_token = use_cls_token

        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        cls, feat_map = self.backbone(x)
        if self.use_cls_token:
            feat = cls
        else:
            feat = feat_map.mean(dim=[2, 3])  # global average pooling
        logits = self.head(feat)
        return logits


# --------------------------------------------------
# 下游任務 2：語意分割（像 UNet 最後幾層）
# --------------------------------------------------

class CellSegmenter(nn.Module):
    """
    非常簡單版的 seg head：
    - 取 backbone feature map (B, C, h, w)
    - 兩層 conv
    - 上採樣回原圖解析度
    """
    def __init__(
        self,
        backbone: CellViTBackbone,
        num_classes: int,
        upsample_factor: int,
    ):
        super().__init__()
        self.backbone = backbone
        dim = self.backbone.encoder.norm.normalized_shape[0]

        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, num_classes, 1)

        self.upsample_factor = upsample_factor

    def forward(self, x):
        cls, feat_map = self.backbone(x)      # (B, C, h, w)
        x = F.relu(self.bn1(self.conv1(feat_map)))
        x = self.conv2(x)
        x = F.interpolate(
            x,
            scale_factor=self.upsample_factor,
            mode="bilinear",
            align_corners=False,
        )
        return x  # (B, num_classes, H, W)


# --------------------------------------------------
# 下游任務 3：簡單密集偵測 head
# --------------------------------------------------

class CellDetector(nn.Module):
    """
    極簡版本的 detector：
    - 在 patch feature map 上預測:
      [objectness / class logits, bbox(dx, dy, dw, dh)] 每個 patch 一個 anchor
    - out: (B, num_classes + 4, h, w)
    之後你可以自己寫 loss 把這個轉成你想要的 DET 損失。
    """
    def __init__(self, backbone: CellViTBackbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        dim = self.backbone.encoder.norm.normalized_shape[0]

        # out_channels = num_classes (分類) + 4 (bbox)
        self.head = nn.Conv2d(dim, num_classes + 4, kernel_size=3, padding=1)

    def forward(self, x):
        cls, feat_map = self.backbone(x)  # (B, C, h, w)
        out = self.head(feat_map)         # (B, num_classes+4, h, w)
        return out


# --------------------------------------------------
# 簡單測試
# --------------------------------------------------

if __name__ == "__main__":
    # 隨便造一張假影像測 pipeline 有沒有爆掉
    imgs = torch.randn(2, 3, 224, 224)

    # 1. 建 MAE 模型（先拿來 pretrain）
    mae = MAE(img_size=224, patch_size=16, in_chans=3)

    # pretrain 階段示意：
    loss, _, _ = mae.forward_loss(imgs)
    print("MAE pretrain loss:", float(loss))

    # 2. 下游任務：建立共用 backbone
    backbone = CellViTBackbone(mae, freeze_encoder=False)

    # 2-1. 分類
    clf = CellClassifier(backbone, num_classes=2)
    logits = clf(imgs)
    print("CLS logits:", logits.shape)  # (B, 2)

    # 2-2. 分割（假設輸入 224、patch 16 -> feature map 14x14 -> upsample_factor=16）
    seg = CellSegmenter(backbone, num_classes=2, upsample_factor=16)
    seg_out = seg(imgs)
    print("SEG out:", seg_out.shape)    # (B, 2, 224, 224)

    # 2-3. 偵測
    det = CellDetector(backbone, num_classes=1)  # 例如只偵測「某種 cell cluster」
    det_out = det(imgs)
    print("DET out:", det_out.shape)    # (B, 1+4, h, w)
