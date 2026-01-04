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

## License

Apache 2.0
