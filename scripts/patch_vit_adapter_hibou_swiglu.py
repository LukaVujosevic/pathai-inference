from pathlib import Path


def main() -> None:
    vit_path = Path(
        "/workspace/ViT-Adapter/segmentation/mmseg_custom/models/backbones/base/vit.py"
    )
    if not vit_path.exists():
        raise SystemExit(f"Missing ViT-Adapter vit.py: {vit_path}")

    source = vit_path.read_text()
    source = source.replace("\\n", "\n")

    if "class SwiGLUFFN" not in source:
        source = source.replace(
            "\n\nclass Block(nn.Module):",
            """

class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Block(nn.Module):""",
        )

    source = source.replace(
        "window_size=14, pad_mode='constant', layer_scale=False, with_cp=False):",
        "window_size=14, pad_mode='constant', layer_scale=False, with_cp=False,\n"
        "                 ffn_type='mlp', ffn_bias=True):",
    )
    source = source.replace(
        "act_layer=nn.GELU, window_attn=False, window_size=14, pretrained=None,\n"
        "                 with_cp=False):",
        "act_layer=nn.GELU, window_attn=False, window_size=14, pretrained=None,\n"
        "                 with_cp=False, ffn_type='mlp', ffn_bias=True):",
    )
    source = source.replace(
        "windowed=window_attn[i], window_size=window_size[i], layer_scale=layer_scale, with_cp=with_cp)",
        "windowed=window_attn[i], window_size=window_size[i], layer_scale=layer_scale, with_cp=with_cp,\n"
        "                ffn_type=ffn_type, ffn_bias=ffn_bias)",
    )

    vit_path.write_text(source)
    print(f"Patched HIBOU SwiGLU support in {vit_path}")


if __name__ == "__main__":
    main()
