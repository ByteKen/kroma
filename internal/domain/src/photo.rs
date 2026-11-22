use crate::develop::DevelopRecipe;
use crate::error::DomainError;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhotoId(String);

impl PhotoId {
    pub fn new(id: impl Into<String>) -> Result<Self, DomainError> {
        let value = id.into();
        if value.trim().is_empty() {
            return Err(DomainError::EmptyIdentifier("PhotoId"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExifTelemetry {
    pub camera_model: String,
    pub lens_model: String,
    pub shutter_speed: String,
    pub aperture_f_number: f32,
    pub iso: u32,
    pub focal_length_mm: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct StarRating(u8);

impl StarRating {
    pub fn new(stars: u8) -> Result<Self, DomainError> {
        if stars > 5 {
            return Err(DomainError::InvalidStarRating(stars));
        }
        Ok(Self(stars))
    }

    pub fn value(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhotoAsset {
    id: PhotoId,
    file_name: String,
    source_path: String,
    capture_timestamp_epoch_sec: i64,
    exif: ExifTelemetry,
    rating: StarRating,
    develop_recipe: DevelopRecipe,
}

impl PhotoAsset {
    pub fn new(
        id: PhotoId,
        file_name: impl Into<String>,
        source_path: impl Into<String>,
        capture_timestamp_epoch_sec: i64,
        exif: ExifTelemetry,
    ) -> Result<Self, DomainError> {
        let file_name = file_name.into();
        if file_name.trim().is_empty() {
            return Err(DomainError::EmptyIdentifier("file_name"));
        }
        let source_path = source_path.into();
        if source_path.trim().is_empty() {
            return Err(DomainError::EmptyIdentifier("source_path"));
        }
        Ok(Self {
            id,
            file_name,
            source_path,
            capture_timestamp_epoch_sec,
            exif,
            rating: StarRating::default(),
            develop_recipe: DevelopRecipe::default(),
        })
    }

    pub fn id(&self) -> &PhotoId {
        &self.id
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub fn source_path(&self) -> &str {
        &self.source_path
    }

    pub fn capture_timestamp_epoch_sec(&self) -> i64 {
        self.capture_timestamp_epoch_sec
    }

    pub fn exif(&self) -> &ExifTelemetry {
        &self.exif
    }

    pub fn rating(&self) -> StarRating {
        self.rating
    }

    pub fn develop_recipe(&self) -> &DevelopRecipe {
        &self.develop_recipe
    }

    pub fn set_rating(&mut self, rating: StarRating) {
        self.rating = rating;
    }

    pub fn replace_develop_recipe(&mut self, recipe: DevelopRecipe) {
        self.develop_recipe = recipe;
    }

    pub fn develop_recipe_mut(&mut self) -> &mut DevelopRecipe {
        &mut self.develop_recipe
    }
}
