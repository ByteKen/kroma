use kroma_domain::error::DomainError;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum AppError {
    Domain(DomainError),
    NotFound(String),
    InvalidInput(String),
    StorageFailure(String),
    ImageProcessingFailure(String),
}

impl From<DomainError> for AppError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::NotFound(msg) => write!(f, "not found: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::StorageFailure(msg) => write!(f, "storage failure: {msg}"),
            Self::ImageProcessingFailure(msg) => write!(f, "image processing failure: {msg}"),
        }
    }
}

impl std::error::Error for AppError {}
