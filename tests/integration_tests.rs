use std::path::Path;
use std::process::Command;

/// Test images and their center points for segmentation
const TEST_CASES: &[(&str, &str, (u32, u32))] = &[
    ("truck", "../sam3/assets/images/truck.jpg", (900, 600)),
    ("groceries", "../sam3/assets/images/groceries.jpg", (400, 300)),
    ("test_image", "../sam3/assets/images/test_image.jpg", (320, 240)),
    ("packraft", "../../packraft.jpg", (720, 720)),
];

const VISION_MODEL: &str = "../models/sam3-tracker-ONNX/onnx/vision_encoder_q4.onnx";
const DECODER_MODEL: &str = "../models/sam3-tracker-ONNX/onnx/prompt_encoder_mask_decoder_q4.onnx";

fn get_binary_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop(); // Remove test binary name
    path.pop(); // Remove deps
    path.push("sam3-rust");
    path
}

#[test]
fn test_models_exist() {
    let vision_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(VISION_MODEL);
    let decoder_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DECODER_MODEL);

    assert!(vision_path.exists(), "Vision model not found at {:?}", vision_path);
    assert!(decoder_path.exists(), "Decoder model not found at {:?}", decoder_path);
}

#[test]
fn test_truck_segmentation() {
    run_segmentation_test("truck", "../sam3/assets/images/truck.jpg", 900, 600);
}

#[test]
fn test_groceries_segmentation() {
    run_segmentation_test("groceries", "../sam3/assets/images/groceries.jpg", 400, 300);
}

#[test]
fn test_test_image_segmentation() {
    run_segmentation_test("test_image", "../sam3/assets/images/test_image.jpg", 320, 240);
}

#[test]
fn test_packraft_segmentation() {
    run_segmentation_test("packraft", "../../packraft.jpg", 720, 720);
}

fn run_segmentation_test(name: &str, image_path: &str, x: u32, y: u32) {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let binary = get_binary_path();

    // Ensure binary exists (run cargo build first)
    if !binary.exists() {
        panic!("Binary not found at {:?}. Run 'cargo build' first.", binary);
    }

    let image_full_path = manifest_dir.join(image_path);
    assert!(image_full_path.exists(), "Test image not found at {:?}", image_full_path);

    let output_path = std::env::temp_dir().join(format!("sam3_test_{}.png", name));

    let output = Command::new(&binary)
        .current_dir(manifest_dir)
        .arg("--vision-model")
        .arg(VISION_MODEL)
        .arg("--decoder-model")
        .arg(DECODER_MODEL)
        .arg("--image")
        .arg(image_path)
        .arg("--point")
        .arg(format!("{},{}", x, y))
        .arg("--mask-index")
        .arg("1")
        .arg("--output")
        .arg(&output_path)
        .output()
        .expect("Failed to execute sam3-rust");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("=== {} ===", name);
    println!("stdout:\n{}", stdout);
    if !stderr.is_empty() {
        println!("stderr:\n{}", stderr);
    }

    assert!(output.status.success(), "Segmentation failed for {}: {}", name, stderr);
    assert!(output_path.exists(), "Output mask not created for {}", name);

    // Verify mask file is valid
    let metadata = std::fs::metadata(&output_path).unwrap();
    assert!(metadata.len() > 0, "Output mask is empty for {}", name);

    // Check for IoU scores in output
    assert!(stdout.contains("IoU ="), "No IoU scores in output for {}", name);
    assert!(stdout.contains("Object presence probability"), "No object score in output for {}", name);

    // Clean up
    let _ = std::fs::remove_file(&output_path);

    println!("{}: PASSED\n", name);
}

#[test]
fn test_box_prompt() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let binary = get_binary_path();

    if !binary.exists() {
        panic!("Binary not found. Run 'cargo build' first.");
    }

    let output_path = std::env::temp_dir().join("sam3_test_box.png");

    let output = Command::new(&binary)
        .current_dir(manifest_dir)
        .arg("--vision-model")
        .arg(VISION_MODEL)
        .arg("--decoder-model")
        .arg(DECODER_MODEL)
        .arg("--image")
        .arg("../sam3/assets/images/truck.jpg")
        .arg("--box")
        .arg("200,100,1600,1000")
        .arg("--output")
        .arg(&output_path)
        .output()
        .expect("Failed to execute sam3-rust");

    assert!(output.status.success(), "Box prompt segmentation failed");
    assert!(output_path.exists(), "Output mask not created for box prompt");

    let _ = std::fs::remove_file(&output_path);

    println!("Box prompt test: PASSED");
}

#[test]
fn test_multiple_points() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let binary = get_binary_path();

    if !binary.exists() {
        panic!("Binary not found. Run 'cargo build' first.");
    }

    let output_path = std::env::temp_dir().join("sam3_test_multi.png");

    let output = Command::new(&binary)
        .current_dir(manifest_dir)
        .arg("--vision-model")
        .arg(VISION_MODEL)
        .arg("--decoder-model")
        .arg(DECODER_MODEL)
        .arg("--image")
        .arg("../sam3/assets/images/truck.jpg")
        .arg("--point")
        .arg("500,300")
        .arg("--point")
        .arg("600,400")
        .arg("--output")
        .arg(&output_path)
        .output()
        .expect("Failed to execute sam3-rust");

    assert!(output.status.success(), "Multiple points segmentation failed");
    assert!(output_path.exists(), "Output mask not created for multiple points");

    let _ = std::fs::remove_file(&output_path);

    println!("Multiple points test: PASSED");
}
