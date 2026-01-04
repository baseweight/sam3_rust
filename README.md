# SAM3 Rust

A Rust implementation for [Segment Anything 3 (SAM3)](https://github.com/facebookresearch/sam3) inference using ONNX Runtime.

## Features

- Vision encoder + prompt encoder/mask decoder pipeline
- Point prompts (`--point x,y`) for click-based segmentation
- Box prompts (`--box x1,y1,x2,y2`) for region-based segmentation
- Multi-mask output with IoU confidence scores
- Cross-platform support:
  - Linux (CPU)
  - Windows (DirectML)
  - macOS (CoreML)

## Requirements

- Rust 1.85+ (edition 2024)
- SAM3 ONNX models (vision encoder + prompt encoder/mask decoder)

## Building

```bash
cargo build --release
```

## Model Setup

Download SAM3 ONNX models from HuggingFace. The recommended quantized models are:
- `vision_encoder_q4.onnx` (~350MB)
- `prompt_encoder_mask_decoder_q4.onnx` (~8MB)

## Usage

### Point Prompt

Click on an object to segment it:

```bash
./target/release/sam3-rust \
  --vision-model path/to/vision_encoder_q4.onnx \
  --decoder-model path/to/prompt_encoder_mask_decoder_q4.onnx \
  --image input.jpg \
  --point 500,300 \
  --output mask.png
```

### Multiple Points

Use multiple points to refine the selection:

```bash
./target/release/sam3-rust \
  --vision-model path/to/vision_encoder_q4.onnx \
  --decoder-model path/to/prompt_encoder_mask_decoder_q4.onnx \
  --image input.jpg \
  --point 500,300 \
  --point 520,350 \
  --output mask.png
```

### Box Prompt

Draw a bounding box around the object:

```bash
./target/release/sam3-rust \
  --vision-model path/to/vision_encoder_q4.onnx \
  --decoder-model path/to/prompt_encoder_mask_decoder_q4.onnx \
  --image input.jpg \
  --box 200,100,800,500 \
  --output mask.png
```

### Mask Selection

SAM3 outputs 3 masks with different granularity. Use `--mask-index` to select:
- `0` - Smallest/most focused mask
- `1` - Medium mask (often best quality)
- `2` - Largest/most inclusive mask

```bash
./target/release/sam3-rust \
  --vision-model path/to/vision_encoder_q4.onnx \
  --decoder-model path/to/prompt_encoder_mask_decoder_q4.onnx \
  --image input.jpg \
  --point 500,300 \
  --mask-index 1 \
  --output mask.png
```

## CLI Options

```
Options:
      --vision-model <PATH>   Path to vision encoder ONNX model
      --decoder-model <PATH>  Path to prompt encoder + mask decoder ONNX model
  -i, --image <PATH>          Path to input image
      --point <X,Y>           Point prompt (can be specified multiple times)
      --box <X1,Y1,X2,Y2>     Box prompt
      --mask-index <N>        Which mask to output: 0, 1, or 2 [default: 0]
  -o, --output <PATH>         Output mask path [default: mask.png]
  -h, --help                  Print help
  -V, --version               Print version
```

## Output

The tool outputs:
- A binary mask PNG image (white = segmented region, black = background)
- IoU confidence scores for each of the 3 masks
- Object presence probability

Example output:
```
Segmentation scores:
  Mask 0: IoU = 0.1209
  Mask 1: IoU = 0.9547
  Mask 2: IoU = 0.7471
  Object presence probability: 1.0000
```

## Prompting System

SAM3 uses a flexible prompting system that allows users to specify what to segment through points, boxes, or combinations.

### How Prompting Works

1. **Vision Encoder** processes the entire image into multi-scale feature embeddings (288x288, 144x144, 72x72)
2. **Prompt Encoder** converts user prompts (points/boxes) into prompt embeddings
3. **Mask Decoder** combines image features + prompt embeddings to predict segmentation masks

### Point Prompts

Points are the simplest prompt type - click on an object to segment it.

| Property | Description |
|----------|-------------|
| Format | `(x, y)` coordinates in original image space |
| Label | `1` = foreground (include), `0` = background (exclude) |
| Tensor | `input_points` [batch, 1, num_points, 2] float32 |
| Labels Tensor | `input_labels` [batch, 1, num_points] int64 |

**Use cases:**
- Single click to segment an object
- Multiple foreground points to ensure full object coverage
- Background points to exclude regions (e.g., click on background to say "not this")

### Box Prompts

Boxes define a region where the object is located.

| Property | Description |
|----------|-------------|
| Format | `(x1, y1, x2, y2)` - top-left and bottom-right corners |
| Tensor | `input_boxes` [batch, num_boxes, 4] float32 |

**Use cases:**
- When object boundaries are roughly known
- Combined with points for more precise segmentation
- Multiple boxes for multi-object segmentation

### Multi-Mask Output

SAM3 outputs 3 masks for each prompt to handle ambiguity:

| Mask Index | Description | Example |
|------------|-------------|---------|
| 0 | Smallest/most specific | Just a person's hand |
| 1 | Medium granularity | The whole person |
| 2 | Largest/most inclusive | Person + their shadow |

Each mask includes an **IoU score** (0-1) indicating model confidence. Higher IoU = more confident.

### Coordinate System

- Origin `(0, 0)` is **top-left** of the image
- X increases rightward, Y increases downward
- Coordinates are in **original image space** (automatically scaled to model's 1008x1008 internally)

### Prompt Combinations

| Combination | Quality | Use Case |
|-------------|---------|----------|
| Single point | Good | Quick selection |
| Multiple points | Better | Complex objects |
| Box only | Good | Known bounding box |
| Point + Box | Best | Precise segmentation |
| Points with labels | Best | Include/exclude regions |

## Integration Notes

This PoC is designed for integration into [Baseweight Canvas](https://github.com/baseweight/canvas).

### API Surface for UI Integration

The core segmentation can be called programmatically:

```rust
use sam3_rust::{Sam3, Sam3ImageProcessor};

// Load models once
let mut model = Sam3::new(&vision_model_path, &decoder_model_path)?;

// For each segmentation request:
let image = image::open(&image_path)?;
let points = vec![(500.0, 300.0)];  // From UI click
let labels = vec![1i64];             // Foreground
let bbox = None;                     // Optional box from UI drag

let result = model.segment(image, &points, &labels, bbox)?;

// result.masks: [batch, num_prompts, 3, H, W] - three mask options
// result.iou_scores: [batch, num_prompts, 3] - confidence per mask
// result.object_scores: [batch, num_prompts, 1] - object presence
```

### UI Interaction Patterns

| User Action | Prompt Type | Implementation |
|-------------|-------------|----------------|
| Click | Point (foreground) | `--point x,y` |
| Shift+Click | Point (background) | Add `--bg-point x,y` (TODO) |
| Drag rectangle | Box | `--box x1,y1,x2,y2` |
| Click + Drag | Point + Box | Combined prompts |

### Recommended UI Flow

1. User loads image
2. Run vision encoder **once** (cache embeddings)
3. User clicks/drags to add prompts
4. Run prompt encoder + mask decoder (fast, ~50ms)
5. Display all 3 masks with IoU scores
6. User selects preferred mask or refines prompts

### VLM Integration

SAM3 can be paired with Vision-Language Models for text-prompted segmentation:

1. **VLM** (e.g., SmolVLM) identifies object locations from text: "Find the red car"
2. VLM outputs bounding box or point coordinates
3. **SAM3** segments the identified region precisely

This enables natural language segmentation: "Segment the person on the left"

### Performance Considerations

| Stage | Time (CPU) | Cacheable |
|-------|------------|-----------|
| Vision Encoder | ~10-15s | Yes (per image) |
| Prompt Encoder + Mask Decoder | ~50-100ms | No (per prompt) |

**Optimization**: Cache vision encoder output when user is iteratively refining prompts on the same image.

## Future Work

- [ ] Background point support (`--bg-point`)
- [ ] Auto-select best mask (`--best-mask`)
- [ ] Batch processing multiple images
- [ ] Video tracking mode
- [ ] Library API for programmatic use
- [ ] GPU acceleration on Linux (CUDA)

## License

Apache 2.0
