# DINOv2 for Semantic Segmentation

This project implements semantic segmentation using Facebook's DINOv2 vision transformer as the backbone. The implementation focuses on person segmentation but can be extended to other classes.

## Architecture Design Choices

### Backbone: DINOv2
- Using DINOv2's small variant (`vits14`) as the backbone
- Leverages self-supervised pre-training for better feature extraction
- Benefits from DINOv2's strong representation learning capabilities

### Segmentation Head
- Custom segmentation head with BatchNorm2d for feature normalization
- Multi-scale feature fusion from different transformer layers
- Final 1x1 convolution for pixel-wise classification

### Key Design Decisions
1. **Multi-scale Features**: Using features from multiple transformer layers (indices [0,1,2,3]) for better segmentation
2. **BatchNorm**: Using BatchNorm2d instead of SyncBatchNorm for better compatibility across devices
3. **Device Support**: Optimized for both CPU and GPU (CUDA/MPS) with automatic device fallback

## DINOv2 Integration

### How DINOv2 is Used
1. **Backbone Initialization**:
   ```python
   backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
   ```

2. **Feature Extraction**:
   - Modified forward pass to extract intermediate features
   - Uses `get_intermediate_layers` for multi-scale feature extraction

3. **Integration with Segmentation Head**:
   - Features from DINOv2 are processed through a custom segmentation head
   - Maintains spatial information through proper reshaping and upsampling

## Training and Performance

### Training Configuration
- Optimizer: AdamW
- Learning Rate: 1e-4
- Batch Size: 8
- Image Size: 640x640
- Dataset: Custom person segmentation dataset

### Performance Metrics and Results
The model's performance is tracked through several metrics:

1. **Training Loss** (`outputs/loss.png`):
   - Tracks the training loss over epochs
   - Shows convergence of the model

2. **Accuracy** (`outputs/accuracy.png`):
   - Monitors pixel-wise accuracy
   - Helps identify overfitting

3. **Mean IoU** (`outputs/miou.png`):
   - Intersection over Union metric
   - Key metric for segmentation quality

### Inference Performance
- Real-time inference on both images and videos
- Average FPS: ~65
- Supports both CPU and GPU acceleration

## Sample Results

### Segmentation Visualization
The model produces segmentation masks with the following classes:
- Background (Black)
- Person (Green)

Sample results can be found in:
- Images: `outputs/inference_results_image/`
- Videos: `outputs/inference_results_video/`
- Validation predictions: `outputs/valid_preds/`

## Setup and Usage

### Prerequisites
- Python 3.9+
- PyTorch 2.5.1
- CUDA (optional) 

### Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

1. **Modelling**:
   ```bash
   python model.py
   ```
   
2. **Training**:
   ```bash
   python train.py --imgsz 640 640
   ```

3. **Image Inference**:
   ```bash
   python infer_image.py --imgsz 640 640 --model outputs/best_model_iou.pth
   ```

4. **Video Inference**:
   ```bash
   python infer_video.py --imgsz 640 640 --model outputs/best_model_iou.pth --input path/to/video.mp4
   ```

### Device Support

The code automatically detects and uses the best available device:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon (M1/M2/M3)
- CPU as fallback

For Apple Silicon users, enable MPS fallback for unsupported operations:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python infer_video.py ...
```

## Project Structure

```
.
├── config.py           # Configuration and class definitions
├── datasets.py         # Dataset loading and preprocessing
├── engine.py          # Training and validation loops
├── infer_image.py     # Image inference script
├── infer_video.py     # Video inference script
├── metrics.py         # Evaluation metrics
├── model.py           # DINOv2 model implementation
├── train.py           # Training script
└── utils.py           # Utility functions
```

## License

This project uses DINOv2 which is subject to its own license. Please refer to the [DINOv2 repository](https://github.com/facebookresearch/dinov2) for more information.

## Acknowledgments

- Facebook Research for DINOv2
- PyTorch team for MPS support
- OpenCV for image processing utilities 
