import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


def _ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure the tensor has exactly three channels (C, H, W) by repeating or truncating channels when needed.
    """
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    channels = tensor.shape[0]
    if channels == 3:
        return tensor
    if channels == 1:
        return tensor.repeat(3, 1, 1)
    if channels == 2:
        return torch.cat([tensor, tensor[:1]], dim=0)
    if channels > 3:
        return tensor[:3]
    raise ValueError(f"Unsupported number of channels: {channels}")


def build_image_transform(
    resolution: int,
    center_crop: bool,
    random_flip: bool,
    preserve_input_precision: bool,
):
    spatial = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
        transforms.RandomHorizontalFlip() if random_flip else transforms.RandomHorizontalFlip(p=0.0),
    ]

    if preserve_input_precision:
        pipeline = [
            transforms.PILToTensor(),
            transforms.Lambda(_ensure_three_channels),
            transforms.ConvertImageDtype(torch.float32),
        ] + spatial + [transforms.Normalize([0.5], [0.5])]
    else:
        pipeline = spatial + [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]

    return transforms.Compose(pipeline)


@dataclass
class ManifestRecord:
    path: Path
    metadata: Dict


class ConditionEncoder:
    """
    Turns metadata dicts into dense condition vectors, padding missing keys with zeros.
    Default spec supports: temperature (scalar), wind (vector), wind_mag (scalar).
    Spec format: List of (name, dim). Can be extended later.
    """

    def __init__(self, spec: Optional[List[Tuple[str, int]]] = None):
        if spec is None:
            spec = [("temperature", 1), ("wind", 3), ("wind_mag", 1)]
        self.spec = spec
        self.cond_dim = sum(dim for _, dim in self.spec)

    def encode(self, metadata: Dict) -> torch.Tensor:
        if self.cond_dim == 0:
            return torch.zeros(0)

        values: List[float] = []
        for name, dim in self.spec:
            raw = metadata.get(name)

            if name == "wind_mag" and raw is None:
                wind = metadata.get("wind")
                if isinstance(wind, (list, tuple)):
                    raw = float(sum(v * v for v in wind) ** 0.5)

            if raw is None:
                values.extend([0.0] * dim)
                continue

            if isinstance(raw, (list, tuple)):
                # Flatten list/tuple up to dim
                padded = list(raw)[:dim]
                padded.extend([0.0] * (dim - len(padded)))
                values.extend([float(v) for v in padded])
            else:
                try:
                    scalar = float(raw)
                except (TypeError, ValueError):
                    scalar = 0.0
                values.append(scalar)
                if dim > 1:
                    values.extend([0.0] * (dim - 1))

        return torch.tensor(values, dtype=torch.float32)


def parse_condition_spec(spec_str: Optional[str]) -> List[Tuple[str, int]]:
    """
    Parse a spec string like "temperature:1,wind:3,wind_mag:1" into a list of tuples.
    Empty or "none" returns an empty list.
    """
    if spec_str is None:
        return [("temperature", 1), ("wind", 3), ("wind_mag", 1)]
    spec_str = spec_str.strip()
    if not spec_str or spec_str.lower() == "none":
        return []

    items: List[Tuple[str, int]] = []
    for part in spec_str.split(","):
        name_dim = part.strip()
        if not name_dim:
            continue
        if ":" not in name_dim:
            raise ValueError(f"Condition spec entry must be name:dim, got '{name_dim}'")
        name, dim_str = name_dim.split(":", 1)
        dim = int(dim_str)
        if dim < 1:
            raise ValueError(f"Condition dimension must be >=1 for '{name}'")
        items.append((name.strip(), dim))
    return items


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        transform,
        condition_encoder: Optional[ConditionEncoder],
        limit: Optional[int] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.transform = transform
        self.condition_encoder = condition_encoder
        self.limit = limit

        self.root, self.samples = self._load_manifest()
        if not self.samples:
            raise ValueError(f"No samples found for split '{split}' in {manifest_path}")

    def _load_manifest(self) -> tuple[Path, List[ManifestRecord]]:
        set_record = None
        samples: List[ManifestRecord] = []

        with self.manifest_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_type = record.get("record_type")
                if record_type == "set":
                    set_record = record
                    continue
                if record_type != "sample":
                    continue

                if set_record is None:
                    raise ValueError("Manifest missing top-level 'set' record before samples.")

                record_split = record.get("metadata", {}).get("split")
                if self.split and record_split != self.split:
                    continue

                image_path = Path(set_record["output_root"]) / record["output_rel"]
                if not image_path.exists():
                    logger.warning("Skipping missing image: %s", image_path)
                    continue

                samples.append(ManifestRecord(path=image_path, metadata=record.get("metadata", {})))
                if self.limit and len(samples) >= self.limit:
                    break

        if set_record is None:
            raise ValueError("Manifest must start with a 'set' record.")

        return Path(set_record["output_root"]), samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            image = img.copy()

        if self.transform is not None:
            image = self.transform(image)

        if self.condition_encoder is None:
            cond = torch.zeros(0)
        else:
            cond = self.condition_encoder.encode(sample.metadata)

        return {"pixel_values": image, "cond": cond}
