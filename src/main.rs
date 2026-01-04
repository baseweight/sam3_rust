mod image_processor;

use anyhow::{Context, Result};
use clap::Parser;
use image::DynamicImage;
use ndarray::{Array2, Array3, Array4, ArrayD};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(target_os = "windows")]
use ort::execution_providers::DirectMLExecutionProvider;

#[cfg(target_os = "macos")]
use ort::execution_providers::CoreMLExecutionProvider;

use crate::image_processor::Sam3ImageProcessor;

#[derive(Parser, Debug)]
#[command(author = "SAM3", version = "0.1.0", about = "SAM3 Rust inference", long_about = None)]
struct Args {
    /// Path to the vision encoder ONNX model
    #[arg(long, default_value = "models/sam3-tracker-ONNX/onnx/vision_encoder_q4.onnx")]
    vision_model: PathBuf,

    /// Path to the prompt encoder + mask decoder ONNX model
    #[arg(long, default_value = "models/sam3-tracker-ONNX/onnx/prompt_encoder_mask_decoder_q4.onnx")]
    decoder_model: PathBuf,

    /// Path to the input image
    #[arg(short, long)]
    image: PathBuf,

    /// Point prompts in format "x,y" (can be specified multiple times)
    /// Use positive points to indicate foreground
    #[arg(long = "point", value_parser = parse_point)]
    points: Vec<(f32, f32)>,

    /// Box prompt in format "x1,y1,x2,y2"
    #[arg(long = "box", value_parser = parse_box)]
    bbox: Option<(f32, f32, f32, f32)>,

    /// Output path for the mask image
    #[arg(short, long, default_value = "mask.png")]
    output: PathBuf,

    /// Select which mask to output (0, 1, or 2). SAM3 outputs 3 masks with different granularity.
    #[arg(long, default_value = "0")]
    mask_index: usize,
}

fn parse_point(s: &str) -> Result<(f32, f32), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err("Point must be in format 'x,y'".to_string());
    }
    let x = parts[0].trim().parse::<f32>().map_err(|e| e.to_string())?;
    let y = parts[1].trim().parse::<f32>().map_err(|e| e.to_string())?;
    Ok((x, y))
}

fn parse_box(s: &str) -> Result<(f32, f32, f32, f32), String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        return Err("Box must be in format 'x1,y1,x2,y2'".to_string());
    }
    let x1 = parts[0].trim().parse::<f32>().map_err(|e| e.to_string())?;
    let y1 = parts[1].trim().parse::<f32>().map_err(|e| e.to_string())?;
    let x2 = parts[2].trim().parse::<f32>().map_err(|e| e.to_string())?;
    let y2 = parts[3].trim().parse::<f32>().map_err(|e| e.to_string())?;
    Ok((x1, y1, x2, y2))
}

struct Sam3 {
    vision_session: Session,
    decoder_session: Session,
    image_processor: Sam3ImageProcessor,
}

impl Sam3 {
    fn new(vision_model_path: &PathBuf, decoder_model_path: &PathBuf) -> Result<Self> {
        // Check GPU availability and create sessions with platform-specific execution providers
        #[cfg(target_os = "windows")]
        {
            let provider = DirectMLExecutionProvider::default();
            match provider.is_available() {
                Ok(true) => println!("DirectML is available"),
                Ok(false) => println!("DirectML not available, using CPU"),
                Err(e) => println!("Error checking DirectML: {}", e),
            }
        }

        #[cfg(target_os = "linux")]
        {
            println!("Using CPU execution provider");
        }

        #[cfg(target_os = "macos")]
        {
            let provider = CoreMLExecutionProvider::default();
            match provider.is_available() {
                Ok(true) => println!("CoreML is available"),
                Ok(false) => println!("CoreML not available, using CPU"),
                Err(e) => println!("Error checking CoreML: {}", e),
            }
        }

        // Create vision session
        #[cfg(target_os = "windows")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        #[cfg(target_os = "linux")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        #[cfg(target_os = "macos")]
        let vision_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(vision_model_path)
            .context("Failed to create vision session")?;

        // Create decoder session
        #[cfg(target_os = "windows")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        #[cfg(target_os = "linux")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        #[cfg(target_os = "macos")]
        let decoder_session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(decoder_model_path)
            .context("Failed to create decoder session")?;

        // Print model info
        println!("Vision encoder inputs:");
        for input in &vision_session.inputs {
            println!("  {}: {:?}", input.name, input.input_type);
        }
        println!("Vision encoder outputs:");
        for output in &vision_session.outputs {
            println!("  {}: {:?}", output.name, output.output_type);
        }

        println!("\nDecoder inputs:");
        for input in &decoder_session.inputs {
            println!("  {}: {:?}", input.name, input.input_type);
        }
        println!("Decoder outputs:");
        for output in &decoder_session.outputs {
            println!("  {}: {:?}", output.name, output.output_type);
        }

        let image_processor = Sam3ImageProcessor::new();

        Ok(Self {
            vision_session,
            decoder_session,
            image_processor,
        })
    }

    fn segment(
        &mut self,
        image: DynamicImage,
        points: &[(f32, f32)],
        labels: &[i64],
        bbox: Option<(f32, f32, f32, f32)>,
    ) -> Result<SegmentationResult> {
        // Preprocess image
        println!("Preprocessing image...");
        let (pixel_values, original_size, _resized_size) = self.image_processor.preprocess(image)?;
        println!("  Original size: {:?}", original_size);
        println!("  Pixel values shape: {:?}", pixel_values.shape());

        // Run vision encoder
        println!("Running vision encoder...");
        let mut vision_inputs: HashMap<&str, Value> = HashMap::new();
        vision_inputs.insert("pixel_values", Value::from_array(pixel_values)?.into());

        let vision_outputs = self.vision_session.run(vision_inputs)?;

        // Extract image embeddings
        let image_embeddings_0 = vision_outputs["image_embeddings.0"]
            .try_extract_array::<f32>()?
            .to_owned();
        let image_embeddings_1 = vision_outputs["image_embeddings.1"]
            .try_extract_array::<f32>()?
            .to_owned();
        let image_embeddings_2 = vision_outputs["image_embeddings.2"]
            .try_extract_array::<f32>()?
            .to_owned();

        println!("  Embeddings shapes: {:?}, {:?}, {:?}",
            image_embeddings_0.shape(),
            image_embeddings_1.shape(),
            image_embeddings_2.shape()
        );

        // Transform points to resized image coordinates
        let transformed_points = self.image_processor.transform_points(points, original_size);
        println!("  Transformed points: {:?}", transformed_points);

        // Prepare prompt inputs
        // input_points: [batch, 1, num_points, 2]
        let num_points = transformed_points.len().max(1);
        let mut input_points = Array4::<f32>::zeros((1, 1, num_points, 2));
        let mut input_labels = Array3::<i64>::zeros((1, 1, num_points));

        for (i, (x, y)) in transformed_points.iter().enumerate() {
            input_points[[0, 0, i, 0]] = *x;
            input_points[[0, 0, i, 1]] = *y;
        }
        for (i, &label) in labels.iter().enumerate() {
            input_labels[[0, 0, i]] = label;
        }

        // Prepare box input: [batch, num_boxes, 4]
        let input_boxes = if let Some((x1, y1, x2, y2)) = bbox {
            let transformed = self.image_processor.transform_boxes(&[(x1, y1, x2, y2)], original_size);
            let (tx1, ty1, tx2, ty2) = transformed[0];
            let mut boxes = Array3::<f32>::zeros((1, 1, 4));
            boxes[[0, 0, 0]] = tx1;
            boxes[[0, 0, 1]] = ty1;
            boxes[[0, 0, 2]] = tx2;
            boxes[[0, 0, 3]] = ty2;
            boxes
        } else {
            // No box - use zeros
            Array3::<f32>::zeros((1, 0, 4))
        };

        println!("Running prompt encoder + mask decoder...");
        let mut decoder_inputs: HashMap<&str, Value> = HashMap::new();
        decoder_inputs.insert("input_points", Value::from_array(input_points)?.into());
        decoder_inputs.insert("input_labels", Value::from_array(input_labels)?.into());
        decoder_inputs.insert("input_boxes", Value::from_array(input_boxes)?.into());
        decoder_inputs.insert("image_embeddings.0", Value::from_array(image_embeddings_0)?.into());
        decoder_inputs.insert("image_embeddings.1", Value::from_array(image_embeddings_1)?.into());
        decoder_inputs.insert("image_embeddings.2", Value::from_array(image_embeddings_2)?.into());

        let decoder_outputs = self.decoder_session.run(decoder_inputs)?;

        // Extract outputs
        let iou_scores = decoder_outputs["iou_scores"]
            .try_extract_array::<f32>()?
            .to_owned();
        let pred_masks = decoder_outputs["pred_masks"]
            .try_extract_array::<f32>()?
            .to_owned();
        let object_score_logits = decoder_outputs["object_score_logits"]
            .try_extract_array::<f32>()?
            .to_owned();

        println!("  IoU scores shape: {:?}", iou_scores.shape());
        println!("  Predicted masks shape: {:?}", pred_masks.shape());
        println!("  Object score logits shape: {:?}", object_score_logits.shape());

        Ok(SegmentationResult {
            masks: pred_masks,
            iou_scores,
            object_scores: object_score_logits,
            original_size,
        })
    }
}

struct SegmentationResult {
    masks: ArrayD<f32>,
    iou_scores: ArrayD<f32>,
    object_scores: ArrayD<f32>,
    original_size: (u32, u32),
}

impl SegmentationResult {
    fn get_best_mask(&self) -> Result<Array2<f32>> {
        // Find the mask with highest IoU score
        let iou = &self.iou_scores;
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        // iou_scores shape: [batch, num_prompts, 3]
        for i in 0..3 {
            let score = iou[[0, 0, i]];
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        println!("Selected mask {} with IoU score: {:.4}", best_idx, best_score);
        self.get_mask(best_idx)
    }

    fn get_mask(&self, index: usize) -> Result<Array2<f32>> {
        // masks shape: [batch, num_prompts, num_masks, H, W]
        let shape = self.masks.shape();
        let h = shape[3];
        let w = shape[4];

        let mut mask = Array2::<f32>::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                mask[[y, x]] = self.masks[[0, 0, index, y, x]];
            }
        }
        Ok(mask)
    }

    fn print_scores(&self) {
        println!("\nSegmentation scores:");
        for i in 0..3 {
            let iou = self.iou_scores[[0, 0, i]];
            println!("  Mask {}: IoU = {:.4}", i, iou);
        }
        let object_score = self.object_scores[[0, 0, 0]];
        let object_prob = 1.0 / (1.0 + (-object_score).exp());
        println!("  Object presence probability: {:.4}", object_prob);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate inputs
    if args.points.is_empty() && args.bbox.is_none() {
        anyhow::bail!("At least one point (--point x,y) or box (--box x1,y1,x2,y2) must be specified");
    }

    println!("Loading SAM3 models...");
    let mut model = Sam3::new(&args.vision_model, &args.decoder_model)?;

    println!("\nLoading image: {:?}", args.image);
    let image = image::open(&args.image)
        .context("Failed to open image")?;
    println!("Image size: {}x{}", image.width(), image.height());

    // Prepare labels (all foreground by default)
    let labels: Vec<i64> = vec![1; args.points.len()];

    println!("\nPrompts:");
    for (i, (x, y)) in args.points.iter().enumerate() {
        println!("  Point {}: ({}, {}) - foreground", i, x, y);
    }
    if let Some((x1, y1, x2, y2)) = args.bbox {
        println!("  Box: ({}, {}) to ({}, {})", x1, y1, x2, y2);
    }

    println!("\nRunning segmentation...");
    let result = model.segment(image, &args.points, &labels, args.bbox)?;
    result.print_scores();

    // Get the selected mask
    let mask_index = args.mask_index.min(2);
    println!("\nExtracting mask {}...", mask_index);
    let mask = result.get_mask(mask_index)?;

    // Postprocess and save
    let mask_image = model.image_processor.postprocess_mask(
        &mask.view(),
        result.original_size,
    );

    mask_image.save(&args.output)
        .context("Failed to save mask")?;

    println!("Mask saved to: {:?}", args.output);
    println!("Done!");

    Ok(())
}
