Đây là bản **Đặc tả Kỹ thuật (Technical Specification)** chi tiết


### 📋 PROJECT SPECIFICATION: AI INTERIOR WALL RE-SKINNING

**Role:** You are a Senior Computer Vision & AI Engineer.
**Objective:** Build a Python-based pipeline to change the color/texture of walls in an interior image while preserving furniture, lighting, and shadows (Occlusion Handling).
**Target Output:** A Python script/module that takes a source image + reference color image and outputs the modified image.

---

### 1. SYSTEM ARCHITECTURE

The pipeline consists of 3 main stages:

1. **Segmentation Engine:** Extracts the wall mask using **FastSAM** (Fast Segment Anything Model).
2. **Conditioning Engine:** Prepares Depth Map (via **ControlNet Preprocessor**) and Color Embeddings (via **IP-Adapter**).
3. **Generative Engine:** Uses **Stable Diffusion 1.5 Inpainting** + **ControlNet Depth** + **IP-Adapter** to generate the new wall.

### 2. TECH STACK & DEPENDENCIES

* **Language:** Python 3.10+
* **Core Libraries:** `diffusers`, `torch`, `opencv-python`, `transformers`, `accelerate`, `safetensors`.
* **Specific Models:**
* Base: `runwayml/stable-diffusion-inpainting`
* ControlNet: `lllyasviel/control_v11f1p_sd15_depth`
* Adapter: `huggingface/h94/IP-Adapter` (Model: `ip-adapter_sd15.bin`)
* Segmentation: `Ultralytics` (YOLO-based FastSAM) or `MobileSAM`.



---

### 3. IMPLEMENTATION WORKFLOW (Step-by-Step)

#### Step 1: Environment Setup

Create a `requirements.txt` file with the following:

```text
torch>=2.0.1
torchvision>=0.15.2
diffusers>=0.24.0
transformers>=4.30.0
accelerate>=0.20.0
opencv-python
ultralytics
controlnet_aux
scipy
mediapipe

```

#### Step 2: The Segmentation Module (`segmentation.py`)

* **Goal:** Create a function `get_wall_mask(image_path, text_prompt="wall")`.
* **Logic:**
1. Load FastSAM model (e.g., `FastSAM-x.pt`).
2. Run inference on the input image.
3. Use the text prompt "wall" to filter relevant masks (using CLIP score inherent in FastSAM logic or simple class filtering).
4. Merge all "wall" masks into a single binary mask.
5. **Important:** Dilate the mask slightly (kernel size 5-10) to cover the edges between the wall and furniture to prevent white borders artifacts.



#### Step 3: The Generative Pipeline (`pipeline.py`)

* **Goal:** Create a class `WallReskinPipeline`.
* **Initialization (`__init__`):**
1. Load `StableDiffusionControlNetInpaintPipeline`.
2. Load `ControlNetModel` from `lllyasviel/control_v11f1p_sd15_depth`.
3. Load `IP-Adapter` using `pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")`.
4. Move everything to GPU (`cuda`).
5. Enable `xformers` or `model_cpu_offload` for memory optimization.


* **Execution (`process`):**
1. Input: `source_image`, `mask_image`, `reference_image`.
2. **Pre-processing:**
* Resize `reference_image` to 224x224 (for CLIP encoder).
* Generate `control_image` (Depth Map) from `source_image` using `ControlNetAux` (MidasDetector).


3. **Inference:**
* Call the pipeline with:
* `prompt`: "a wall, high quality, interior design, photorealistic" (Generic prompt is fine because IP-Adapter handles the color).
* `image`: source_image
* `mask_image`: mask_image
* `control_image`: depth_map
* `ip_adapter_image`: reference_image
* `negative_prompt`: "blurry, low quality, artifacts, distortion, furniture changes"
* `strength`: 1.0 (Denoising strength).
* `controlnet_conditioning_scale`: 0.8 (Keep structure strong).
* `ip_adapter_scale`: 0.6 to 0.8 (Adjust based on how strong you want the color).







#### Step 4: Main Execution Script (`main.py`)

* Orchestrate the flow:
1. Load Source Image.
2. Load Reference Color Image.
3. Call `segmentation.get_wall_mask` -> Get Mask.
4. Call `pipeline.process` -> Get Result.
5. Save Result.



---

### 4. CRITICAL CODE SKELETON (For the AI to expand)

Use this skeleton for the `pipeline.py` to ensure the correct usage of Diffusers libraries:

```python
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from controlnet_aux import MidasDetector
from PIL import Image

class WallReskinPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load ControlNet (Depth)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float16
        )

        # 2. Load Main Pipeline (SD 1.5 Inpainting)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        )

        # 3. Load IP-Adapter (For Style/Color Transfer)
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        self.pipe.set_ip_adapter_scale(0.7) # Adjustable: 0.5 - 0.8

        # 4. Load Depth Estimator
        self.depth_estimator = MidasDetector.from_pretrained("lllyasviel/Annotators")

        # Optimization
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload() 

    def process(self, image_path, mask_path, ref_path):
        # Load Images
        init_image = Image.open(image_path).convert("RGB").resize((512, 512))
        mask_image = Image.open(mask_path).convert("RGB").resize((512, 512))
        ref_image = Image.open(ref_path).convert("RGB").resize((224, 224)) # IP-Adapter needs this size

        # Prepare Depth Map
        depth_map = self.depth_estimator(init_image)

        # Generate
        generator = torch.Generator(device=self.device).manual_seed(42)
        result = self.pipe(
            prompt="high quality wall", 
            image=init_image,
            mask_image=mask_image,
            control_image=depth_map,
            ip_adapter_image=ref_image,
            num_inference_steps=30,
            generator=generator,
            controlnet_conditioning_scale=0.8,
            strength=0.99
        ).images[0]

        return result

```

### 5. INSTRUCTIONS FOR IDE

1. **Analyze** the architecture above.
2. **Create** the file structure.
3. **Write** the `segmentation.py` using Ultralytics FastSAM.
4. **Complete** the `pipeline.py` based on the skeleton.
5. **Create** a `main.py` to run a test case.

---

