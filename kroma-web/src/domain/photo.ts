import type { DevelopRecipe } from './develop';

export interface ExifTelemetry {
  readonly shutterSpeed: string;
  readonly aperture: string;
  readonly iso: number;
  readonly focalLengthMm: string;
  readonly cameraModel: string;
  readonly lensModel: string;
}

export interface Point2DPercent {
  readonly x: number;
  readonly y: number;
}

/** YuNet bounding box & 5-point landmarks in normalized [0..100] image percentages. */
export interface YuNetFaceOverlay {
  readonly topPct: number;
  readonly leftPct: number;
  readonly widthPct: number;
  readonly heightPct: number;
  readonly rightEye: Point2DPercent;
  readonly leftEye: Point2DPercent;
  readonly noseTip: Point2DPercent;
  readonly rightMouth: Point2DPercent;
  readonly leftMouth: Point2DPercent;
}

export interface PhotoAsset {
  readonly id: string;
  readonly fileName: string;
  readonly directoryId: string;
  readonly virtualFolderId?: string | null;
  readonly imageUrl: string;
  /** Optional 32-bit Rust/wgpu rendered JPEG thumbnail matching `developRecipe` 1:1. */
  readonly renderedThumbnailUrl?: string | null;
  readonly fileSizeBytes?: number;
  readonly isSlateAnchor: boolean;
  readonly clusterId: string | null;
  /** SFace cosine similarity in [0.0, 1.0]. Slate anchors default to 1.0. */
  readonly sfaceCosineSimilarity: number;
  readonly capturedAtEpochSec?: number | null;
  readonly capturedAtLabel?: string | null;
  readonly sharpnessPct: number;
  readonly eyeAfLocked: boolean;
  readonly heroSelect: boolean;
  readonly exif: ExifTelemetry;
  readonly faceOverlay: YuNetFaceOverlay | null;
  readonly developRecipe: DevelopRecipe;
}
