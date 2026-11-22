use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum DomainError {
    EmptyIdentifier(&'static str),
    ZeroMagnitudeEmbedding,
    InvalidSimilarityThreshold(f32),
    InvalidStarRating(u8),
    InvalidColorTemperature(u32),
    InvalidCropRect { left: f32, top: f32, width: f32, height: f32 },
    InvalidLutIntensity(f32),
    EmptyOutputTemplate,
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(field) => write!(f, "{field} cannot be empty"),
            Self::ZeroMagnitudeEmbedding => {
                write!(f, "face embedding vector cannot have zero magnitude")
            }
            Self::InvalidSimilarityThreshold(val) => {
                write!(f, "similarity threshold {val} must be within [0.0, 1.0]")
            }
            Self::InvalidStarRating(stars) => {
                write!(f, "star rating {stars} must be between 0 and 5")
            }
            Self::InvalidColorTemperature(kelvin) => {
                write!(f, "color temperature {kelvin}K must be within [2000, 50000]")
            }
            Self::InvalidCropRect {
                left,
                top,
                width,
                height,
            } => write!(
                f,
                "invalid normalized crop rect (left={left}, top={top}, width={width}, height={height})"
            ),
            Self::InvalidLutIntensity(intensity) => {
                write!(f, "LUT intensity {intensity} must be within [0.0, 1.0]")
            }
            Self::EmptyOutputTemplate => write!(f, "output naming template cannot be empty"),
        }
    }
}

impl std::error::Error for DomainError {}
