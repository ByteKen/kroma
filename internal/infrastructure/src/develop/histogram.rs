use image::RgbImage;
use kroma_domain::develop::RgbHistogram;

/// Computes a 256-bin RGB + Rec.709 luminance histogram for darkroom telemetry panels.
pub fn compute_rgb_histogram(image: &RgbImage) -> RgbHistogram {
    let mut red = [0_u32; 256];
    let mut green = [0_u32; 256];
    let mut blue = [0_u32; 256];
    let mut luminance = [0_u32; 256];

    for pixel in image.pixels() {
        let [r, g, b] = pixel.0;
        red[r as usize] += 1;
        green[g as usize] += 1;
        blue[b as usize] += 1;
        let luma = (0.2126 * (r as f32) + 0.7152 * (g as f32) + 0.0722 * (b as f32))
            .round()
            .clamp(0.0, 255.0) as usize;
        luminance[luma] += 1;
    }

    let total_pixels = (image.width() as u64) * (image.height() as u64);
    RgbHistogram {
        red,
        green,
        blue,
        luminance,
        total_pixels,
    }
}
