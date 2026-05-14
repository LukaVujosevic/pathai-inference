# Configs

`mask2former_hibou_adapter_large_512_20k_pathtests_1mpp_v1.py` was recovered from the checkpoint metadata of:

```text
best_mean_micro_dice_1_4_iter_8000.pth
```

It is the exact resolved MMSeg config embedded in the checkpoint, not a hand-written approximation.

On RunPod, `scripts/runpod_setup_512.sh` copies it into:

```text
/workspace/ViT-Adapter/segmentation/configs/pathtests/
```

The config still requires ViT-Adapter's custom code and old MMSeg stack to load the model.
