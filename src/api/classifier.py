from __future__ import annotations

import io
from pathlib import Path

from .models import PredictionScore, PreprocessingOptions


class PyTorchFoodClassifier:
    """Lazy-loading PyTorch classifier used by the FastAPI app."""

    def __init__(self, model_path: Path, classes_path: Path) -> None:
        self.model_path = model_path
        self.classes_path = classes_path
        self._model = None
        self._class_names: list[str] = []
        self._torch = None
        self._nn = None
        self._torchvision_models = None
        self._transforms = None
        self._Image = None
        self._UnidentifiedImageError = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        if self._class_names:
            return self._class_names
        if self.classes_path.exists():
            return [
                line.strip()
                for line in self.classes_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return []

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")

        try:
            import torch
            import torch.nn as nn
            from PIL import Image, UnidentifiedImageError
            from torchvision import models as torchvision_models
            from torchvision import transforms
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing ML dependencies. Install the Python requirements before starting the API."
            ) from exc

        self._torch = torch
        self._nn = nn
        self._torchvision_models = torchvision_models
        self._transforms = transforms
        self._Image = Image
        self._UnidentifiedImageError = UnidentifiedImageError

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location="cpu")

        state_dict = checkpoint
        num_classes = len(self.class_names)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            checkpoint_classes = checkpoint.get("class_names") or []
            if checkpoint_classes:
                self._class_names = [str(name) for name in checkpoint_classes]
            num_classes = int(checkpoint.get("num_classes") or len(self.class_names))

        if not self._class_names:
            self._class_names = self.class_names

        if num_classes <= 0:
            raise RuntimeError("No class metadata available for the model checkpoint.")

        model = torchvision_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model

    def classify(
        self, image_bytes: bytes, options: PreprocessingOptions
    ) -> list[PredictionScore]:
        if self._model is None:
            self.load()

        assert self._torch is not None
        assert self._transforms is not None
        assert self._Image is not None
        assert self._UnidentifiedImageError is not None

        try:
            image = self._Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except self._UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

        transform_steps = [
            self._transforms.Resize((options.resize, options.resize)),
        ]
        if options.center_crop:
            transform_steps.append(self._transforms.CenterCrop((options.resize, options.resize)))
        transform_steps.append(self._transforms.ToTensor())
        if options.normalize:
            transform_steps.append(
                self._transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        transform = self._transforms.Compose(transform_steps)
        input_tensor = transform(image).unsqueeze(0)

        with self._torch.no_grad():
            logits = self._model(input_tensor)
            probabilities = self._torch.softmax(logits, dim=1)[0]

        top_k = min(options.top_k, len(self._class_names))
        scores, indices = self._torch.topk(probabilities, k=top_k)

        return [
            PredictionScore(
                label=self._class_names[index.item()],
                confidence=round(score.item(), 6),
            )
            for score, index in zip(scores, indices)
        ]
