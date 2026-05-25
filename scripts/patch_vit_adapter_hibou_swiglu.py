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
        return self.w3(torch.sigmoid(x1) * x1 * x2)


class Block(nn.Module):""",
        )
    source = source.replace(
        "return self.w3(F.silu(x1) * x2)",
        "return self.w3(torch.sigmoid(x1) * x1 * x2)",
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
        "self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)",
        "if ffn_type in ('swiglu', 'swiglufused', 'swiglualigned'):\n"
        "            self.mlp = SwiGLUFFN(in_features=dim, hidden_features=mlp_hidden_dim,\n"
        "                                 act_layer=act_layer, drop=drop, bias=ffn_bias)\n"
        "        else:\n"
        "            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)",
    )
    source = source.replace(
        "windowed=window_attn[i], window_size=window_size[i], layer_scale=layer_scale, with_cp=with_cp)",
        "windowed=window_attn[i], window_size=window_size[i], layer_scale=layer_scale, with_cp=with_cp,\n"
        "                ffn_type=ffn_type, ffn_bias=ffn_bias)",
    )

    if (
        "class SwiGLUFFN" not in source
        or "ffn_type in ('swiglu', 'swiglufused', 'swiglualigned')" not in source
        or "self.mlp = SwiGLUFFN" not in source
    ):
        raise SystemExit("Failed to patch ViT-Adapter SwiGLU support.")

    vit_path.write_text(source)
    print(f"Patched HIBOU SwiGLU support in {vit_path}")
    print("Verified: ViT blocks instantiate SwiGLUFFN for ffn_type='swiglufused'.")


if __name__ == "__main__":
    main()
