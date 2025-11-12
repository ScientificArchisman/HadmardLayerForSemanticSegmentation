
# ResNetModel (DeepLab-style) — Architecture Diagram

```text
Input: (B, 3, H, W)
   │
   ├─ Stem: Conv7×7 s2 → BN → ReLU → MaxPool3×3 s2     →  (B, 64,  H/4,  W/4)
   │
   ├─ layer1 (C2): Bottleneck ×{3/4/8} stride 1        →  (B, 256, H/4,  W/4)
   │
   ├─ layer2 (C3): Bottleneck ×{4/8/36} stride 2       →  (B, 512, H/8,  W/8)
   │
   ├─ layer3 (C4): Bottleneck
   │     • OS=16: stride 2                             →  (B, 1024, H/16, W/16)
   │     • OS=8 : stride 1, dilation 2                 →  (B, 1024, H/8,  W/8)
   │
   │   ┌────────────────────────── AUX HEAD (train only) ──────────────────────────┐
   │   │  Conv3×3 1024→256 → BN → ReLU → Conv1×1 256→K → Upsample → (B, K, H, W)   │
   │   └───────────────────────────────────────────────────────────────────────────┘
   │
   ├─ layer4 (C5): Bottleneck
   │     • OS=16: stride 1, dilation 2                 →  (B, 2048, H/16, W/16)
   │     • OS=8 : stride 1, dilation 4                 →  (B, 2048, H/8,  W/8)
   │
   └─ MAIN HEAD: ASPP + Classifier
        ASPP (in=2048, out=256 per branch)
          ┌────────────────────────────────────────────────────────────────────┐
          │  Branch 1: 1×1 (rate 1)                                            │
          │  Branch 2: 3×3 dilated (rate r1)                                   │
          │  Branch 3: 3×3 dilated (rate r2)                                   │
          │  Branch 4: 3×3 dilated (rate r3)                                   │
          │  Image Pooling: GAP → 1×1 → upsample to x4 size                    │
          └────────────────────────────────────────────────────────────────────┘
        concat → (B, 1280, ·, ·) → 1×1 1280→256 → BN → ReLU → Dropout(0.1)
        → 3×3 256→256 → BN → ReLU → 1×1 256→K
        → Upsample to (H, W)  →  main logits: (B, K, H, W)
```

---

## Output Stride (OS) & ASPP Rates

* **OS=16**

  * Final encoder map: `(B, 2048, H/16, W/16)`
  * ASPP dilation rates: **[1, 6, 12, 18]**
* **OS=8**

  * Final encoder map: `(B, 2048, H/8, W/8)`
  * ASPP dilation rates: **[1, 12, 24, 36]**

---

## Channel Guide (ResNet-50/101/152)

| Stage | Tensor Shape                        | Notes                      |
| ----- | ----------------------------------- | -------------------------- |
| C2    | (B, 256,  H/4,  W/4)                | `layer1` output            |
| C3    | (B, 512,  H/8,  W/8)                | `layer2` output            |
| C4    | (B, 1024, H/16 or H/8, W/16 or W/8) | `layer3` output (**aux**)  |
| C5    | (B, 2048, H/16 or H/8, W/16 or W/8) | `layer4` output (**main**) |

> **K = num_classes**

---

## Loss Usage

* **Training (aux head enabled)**:
  `Loss = CE(main, y) + λ · CE(aux, y)` with **λ ~ 0.3–0.4**
* **Eval/Inference**: use only **main** logits.
