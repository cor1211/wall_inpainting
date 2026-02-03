# AI Interior Wall Re-skinning

Thay đổi màu sắc và texture tường trong ảnh nội thất sử dụng AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)

## 📋 Features

- **Wall Segmentation**: Tự động phát hiện và tạo mask vùng tường
  - Semantic (Mask2Former) - Chính xác nhất
  - CLIP filtering - Hiểu ngữ nghĩa
  - Heuristic - Nhanh, dựa vào vị trí/kích thước
- **Color/Texture Transfer**: Chuyển màu/texture từ ảnh reference
- **3D Structure Preserved**: Giữ nguyên ánh sáng, bóng đổ, perspective
- **Web API**: REST API với FastAPI

## 🚀 Quick Start

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Tải Model Weights

```bash
python download_models.py
# Hoặc tải tất cả HuggingFace models
python download_models.py --all
```

### 3. Chạy Test

```bash
# Thay đổi màu tường sang màu be
python main.py --source room.jpg --color "200,180,160" --output result.png

# Sử dụng ảnh reference
python main.py --source room.jpg --reference texture.jpg --output result.png

# Lưu thêm mask
python main.py --source room.jpg --color "180,200,220" --save-mask
```

## 📁 Project Structure

```
wall_inpainting/
├── main.py              # Script điều phối chính
├── segmentation.py      # Module phân đoạn tường
├── pipeline.py          # Generative pipeline
├── api.py               # FastAPI web service
├── config.py            # Centralized configuration
├── download_models.py   # Script tải models
├── requirements.txt     # Dependencies
├── models/              # Model weights (FastSAM)
├── outputs/             # Generated images
└── tests/               # Test suite
```

## 🖥️ CLI Usage

```bash
# Basic usage với solid color
python main.py --source <image> --color "R,G,B"

# Sử dụng reference image
python main.py --source <image> --reference <ref_image>

# Tuỳ chỉnh parameters
python main.py --source room.jpg --color "200,180,160" \
    --strategy semantic \
    --steps 40 \
    --controlnet-scale 0.8 \
    --ip-scale 0.7 \
    --seed 42

# Xem tất cả options
python main.py --help
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | required | Ảnh nội thất nguồn |
| `--color` | - | Màu target "R,G,B" |
| `--reference` | - | Ảnh reference cho color/texture |
| `--output` | auto | Đường dẫn output |
| `--strategy` | semantic | semantic/clip/heuristic/auto |
| `--steps` | 30 | Số bước inference |
| `--controlnet-scale` | 0.8 | ControlNet strength |
| `--ip-scale` | 0.7 | IP-Adapter strength |
| `--seed` | random | Random seed |
| `--save-mask` | false | Lưu mask |
| `--include-ceiling` | false | Include ceiling trong mask |

## 🌐 API Usage

### Start Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000

# Development mode với auto-reload
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Wall Segmentation
```bash
curl -X POST "http://localhost:8000/segment" \
    -F "image=@room.jpg" \
    -F "strategy=semantic" \
    -o mask.png
```

#### Full Re-skinning
```bash
curl -X POST "http://localhost:8000/process" \
    -F "source=@room.jpg" \
    -F "reference=@color_ref.jpg" \
    -o result.png
```

#### Re-skin with Solid Color
```bash
curl -X POST "http://localhost:8000/process-color" \
    -F "source=@room.jpg" \
    -F "color=200,180,160" \
    -o result.png
```

## ⚙️ System Requirements

- **Python**: 3.10+
- **GPU**: NVIDIA với CUDA (Khuyến nghị VRAM ≥ 8GB)
- **RAM**: 16GB+
- **Disk**: ~10GB cho models

> ⚠️ Pipeline có thể chạy trên CPU nhưng sẽ rất chậm (~5-10 phút/ảnh)

## 🔧 Configuration

Chỉnh sửa `config.py` để thay đổi default parameters:

```python
from config import config

# Thay đổi default parameters
config.pipeline.num_inference_steps = 40
config.pipeline.ip_adapter_scale = 0.8
config.segmentation.default_strategy = "clip"
```

## 🧪 Testing

```bash
# Chạy tests với pytest
pytest tests/ -v

# Chạy manual tests
python tests/test_segmentation.py
python tests/test_pipeline.py
```

## 📝 License

MIT License
