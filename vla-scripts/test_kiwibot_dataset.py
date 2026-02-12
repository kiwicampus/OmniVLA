#!/usr/bin/env python3

import sys
try:
    import PIL.Image
    if not hasattr(PIL.Image, "Resampling"):
        class _Resampling:
            NEAREST = 0
            LANCZOS = 1
            BILINEAR = 2
            BICUBIC = 3
            BOX = 4
            HAMMING = 5
        PIL.Image.Resampling = _Resampling
except Exception:
    pass

import argparse
from pathlib import Path

_OMNIVLA_ROOT = Path(__file__).resolve().parent.parent
if str(_OMNIVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_OMNIVLA_ROOT))


def _load_processor_or_fallback(vla_path):
    """
    Intenta cargar el processor completo (tokenizer + image_processor).
    Si falla (p. ej. PIL.Image.Resampling en Pillow viejo), usa tokenizer mínimo + transform simple.
    Returns (base_tokenizer, image_transform, action_tokenizer).
    """
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        try:
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        except Exception:
            pass
        processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        from prismatic.vla.action_tokenizer import ActionTokenizer
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        return processor.tokenizer, processor.image_processor.apply_transform, action_tokenizer
    except Exception:
        pass  

    print("  (Processor completo no disponible, usando tokenizer + transform mínimos)")
    from transformers import AutoTokenizer
    import torch
    from torchvision import transforms
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    size = 224
    _trans = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def image_transform(pil_image):
        t = _trans(pil_image)
        return t.unsqueeze(0) if t.dim() == 3 else t
    from prismatic.vla.action_tokenizer import ActionTokenizer
    action_tokenizer = ActionTokenizer(tokenizer)
    return tokenizer, image_transform, action_tokenizer


def _show_sample(sample, label):
    """Imprime el contenido de un sample (claves, formas, y un resumen de valores)."""
    print(f"\n--- {label} ---")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={getattr(v, 'dtype', type(v))}")
        elif hasattr(v, "__len__") and not isinstance(v, str):
            print(f"  {k}: len={len(v)}")
        else:
            print(f"  {k}: {v!r}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Crear KiwiBotDataset (OmniVLA) y mostrar primer y último sample.")
    parser.add_argument("--src-dir", type=Path, required=True, help="Path al dataset LeRobot.")
    parser.add_argument("--vla-path", type=str, default="openvla/openvla-7b", help="Modelo HF para cargar processor.")
    args = parser.parse_args()

    if not args.src_dir.is_dir():
        print(f"Error: no existe el directorio {args.src_dir}")
        return 1

    print("Cargando processor (o fallback tokenizer+transform)...")
    from prismatic.vla.datasets import KiwiBotDatasetComplete
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder

    base_tokenizer, image_transform, action_tokenizer = _load_processor_or_fallback(args.vla_path)

    print("Creando dataset en formato OmniVLA (KiwiBotDataset)...")
    dataset = KiwiBotDatasetComplete(
        src_dir=args.src_dir,
        action_tokenizer=action_tokenizer,
        base_tokenizer=base_tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=PurePromptBuilder,
        predict_stop_token=True,
    )

    n = len(dataset)
    print(f"  len(dataset) = {n}")
    if n == 0:
        print("  No hay muestras válidas (episodios con al menos NUM_ACTIONS_CHUNK pasos).")
        return 0

    # Primer episodio
    print("\nObteniendo primer sample (primer episodio)...")
    #first = dataset[0]
    #_show_sample(first, "Primer episodio (dataset[0])")

    print("Obteniendo último sample (dataset[n-1])...")
    last = dataset[944]
    _show_sample(last, "Último episodio (dataset[len-1])")

    return 0


if __name__ == "__main__":
    sys.exit(main())
