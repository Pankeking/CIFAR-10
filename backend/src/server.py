import sys
import re
import base64
import numpy as np
from PIL import Image
from io import BytesIO

from pathlib import Path
from pydantic import BaseModel
import torch
import uvicorn
from fastapi import FastAPI, Query, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.torch_model import TorchModel
from data.data_loader import load_dataset
from nn.math import softmax
from ui.view import _load_tiny_imagenet_class_ids, _load_wnid_to_words


class BaseResponse(BaseModel):
    status: str = "success"
    message: str = ""

class ModelInfo(BaseModel):
    filename: str = ""
    dataset: str = ""
    loss: str = ""
    samples: int = 0
    epochs: int = 0
    ext: str = ""

class ModelListResponse(BaseResponse):
    models: list[ModelInfo] = []

class ImageInfo(BaseModel):
    image: str
    true_label: int

class ImageResponse(BaseResponse):
    image: ImageInfo

class PredictInfo(BaseModel):
    pred_label: str
    true_label: str
    probs: dict[str, float]

class PredictResponse(BaseResponse):
    prediction: PredictInfo

class HealthInfo(BaseModel):
    torch_device: str

class HealthResponse(BaseResponse):
    health: HealthInfo


_model_cache: dict[str, TorchModel] = {}
_dataset_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_classes_cache: dict[str, list[str]] = {}

sys.path.insert(0, str(Path(__file__).parent / "src"))

app = FastAPI(title="ML Model Server", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")

@router.get("/health")
def health() -> HealthResponse:
    """Basic health check."""
    if torch.backends.mps.is_available():
        torch_device = "mps"
    elif torch.cuda.is_available():
        torch_device = "cuda"
    else:
        torch_device = "cpu"
    return HealthResponse(
        status="success",
        message="health check successful",
        health=HealthInfo(torch_device=torch_device)
    )


@router.get("/models", response_model=ModelListResponse)
def list_models(limit: int = Query(20, ge=1, le=100)) -> ModelListResponse:
    """Parsed model metadata from models/."""
    models_dir = Path(__file__).parent.parent / "models"
    if not models_dir.exists():
        return ModelListResponse(status="error", message="no models found")
    
    pattern = re.compile(r'dataset_(.+?)-model_(.+?)-samples_(\d+)-epochs_(\d+)\.(.+)')
    models = []
    for f in models_dir.iterdir():
        if f.suffix not in {".pt", ".pkl"}:
            continue
        match = pattern.match(f.name)
        if match:
            dataset, loss, samples, epochs, ext = match.groups()
            models.append(ModelInfo(
                filename=f.name,
                dataset=dataset,
                loss=loss,
                samples=int(samples),
                epochs=int(epochs),
                ext=f".{ext}"
            ))
    
    return ModelListResponse(status="success", message=f"listed {len(models)} models successfully", models=sorted(models, key=lambda m: m.epochs, reverse=True)[:limit])

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _infer_model_config(dataset_name: str) -> tuple[int, int]:
    """num_classes, C_out from dataset."""
    if "cifar" in dataset_name.lower():
        return 10, 32
    elif "tiny" in dataset_name.lower():
        return 200, 64
    raise ValueError(f"Unknown dataset: {dataset_name}")

def _get_or_load_model(model_filename: str, datasets_root: str) -> tuple[TorchModel, str, list[str]]:
    """Load/cache model + dataset_name + classes."""
    if model_filename in _model_cache:
        print(f"Model {model_filename} found in cache")
        model = _model_cache[model_filename]
        dataset_name = model.dataset_name
        classes = _classes_cache.get(dataset_name, [])
        return model, dataset_name, classes
    
    models_dir = Path(__file__).parent.parent / "models"
    full_path = models_dir / model_filename
    if not full_path.exists():
        raise HTTPException(404, f"Model {model_filename} not found")
    
    dataset_match = re.match(r'dataset_(.+?)-', model_filename)
    dataset_name = dataset_match.group(1) if dataset_match else "tiny_imagenet"
    
    num_classes, C_out = _infer_model_config(dataset_name)
    model = TorchModel(in_channels=3, num_classes=num_classes, base_channels=C_out)
    model.load(str(full_path))
    model.dataset_name = dataset_name
    model.eval()
    model.to(_get_device())
    
    if dataset_name not in _dataset_cache:
        x_train, y_train, x_test, y_test = load_dataset(dataset_name)
        if hasattr(model, "norm_mean") and model.norm_mean is not None:
            x_test = (x_test - model.norm_mean) / model.norm_std
        _dataset_cache[dataset_name] = (x_test, y_test)
    
    if dataset_name == "cifar10":
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        wnids = sorted(_load_tiny_imagenet_class_ids(datasets_root))
        wnid_to_words = _load_wnid_to_words(datasets_root)
        classes = [wnid_to_words.get(wnid, wnid) for wnid in wnids]
    _classes_cache[dataset_name] = classes
    
    _model_cache[model_filename] = model
    return model, dataset_name, classes

@router.get("/predict/{idx}", response_model=PredictResponse)
def predict(idx: int, model_filename: str = Query(..., description="e.g. dataset_tiny_imagenet-model_...pt")) -> PredictResponse:
    """Predict + top-3 probs for test idx."""
    model, dataset_name, classes = _get_or_load_model(model_filename, "datasets")
    x_test, y_test = _dataset_cache[dataset_name]
    if idx >= len(x_test):
        raise HTTPException(400, "Idx out of test range")
    
    x_single = x_test[idx:idx + 1]
    logits = model.predict_logits(x_single)[0]
    probs = softmax(logits[None, :])[0]
    pred_idx = int(np.argmax(probs))
    pred_name = classes[pred_idx].capitalize()
    true_idx = int(y_test[idx])
    true_name = classes[true_idx].capitalize()
    
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_probs = {classes[int(i)]: float(probs[int(i)]) for i in top3_idx}
    
    return PredictResponse(
        status="success",
        message="prediction retrieved successfully",
        prediction=PredictInfo(pred_label=pred_name, true_label=true_name, probs=top3_probs)
    )

@router.get("/image/{idx}", response_model=ImageResponse)
def get_image(idx: int, model_filename: str = Query(..., description="Same as predict")) -> ImageResponse:
    """Base64 PNG of test image (denorm for display)."""
    model, dataset_name, classes = _get_or_load_model(model_filename, "datasets")
    x_test, y_test = _dataset_cache[dataset_name]
    if idx >= len(x_test):
        raise HTTPException(400, "Idx out of test range")
    
    img_norm = x_test[idx]
    img = np.clip((img_norm.transpose(1, 2, 0) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode()
    
    return ImageResponse(status="success", message="image retrieved successfully", image=ImageInfo(image=f"data:image/png;base64,{b64_img}", true_label=int(y_test[idx])))


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
    )
