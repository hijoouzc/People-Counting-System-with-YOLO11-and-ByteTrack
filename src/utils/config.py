"""
Configuration module for the Person-Counting project.
Centralizes all hardcoded parameters, thresholds, and paths.
Provides an ArgumentParser for unified CLI execution.
"""
import os
import argparse
import supervision as sv
from pathlib import Path

class ProjectPaths:
    """Centralized path management for the entire project."""
    # Resolve the project root dynamically (src/utils/config.py -> parents[2] is the root)
    ROOT = Path(__file__).resolve().parents[2]
    
    # Core directories
    SRC_DIR = ROOT / "src"
    DATA_DIR = ROOT / "data"
    MODELS_DIR = ROOT / "models"
    OUTPUTS_DIR = ROOT / "outputs"
    CONFIGS_DIR = ROOT / "configs"
    EXPERIMENTS_DIR = ROOT / "experiments"
    
    # Specific subdirectories
    RAW_DATA_DIR = DATA_DIR / "raw"
    TRAINED_MODELS_DIR = MODELS_DIR / "trained"


class InferenceConfig:
    """Configuration class for inference scripts."""
    
    # Model parameters
    MODEL_V1_PATH = str(ProjectPaths.TRAINED_MODELS_DIR / "HeadDetect_v1.pt")
    MODEL_V2_PATH = str(ProjectPaths.TRAINED_MODELS_DIR / "HeadDetect_v2.pt")
    CONFIDENCE_THRESHOLD = 0.25
    PERSON_CLASS_ID = 0  # Assuming 0 is the class for 'head'
    
    # Counting line geometry (using supervision.Point)
    LINE_START = sv.Point(0, 250)
    LINE_END = sv.Point(1920, 550)
    
    # SAHI specific geometry
    SAHI_LINE_START = sv.Point(0, 250)
    SAHI_LINE_END = sv.Point(1920, 550)
    
    # Tracker configs
    TRACKER_YAML_PATH = str(ProjectPaths.CONFIGS_DIR / "custom_tracker.yaml")
    
    # Visual aesthetics
    COLOR_LINE = (0, 255, 255)       # Yellow
    COLOR_ABOVE = (0, 255, 0)        # Green
    COLOR_BELOW = (0, 165, 255)      # Orange (BGR)
    COLOR_HUD_IN = (0, 255, 0)       # Green
    COLOR_HUD_OUT = (0, 0, 255)      # Red
    COLOR_HUD_INFO = (255, 255, 0)   # Cyan/Yellow
    
    # Default Input/Output
    DEFAULT_VIDEO_PATH = str(ProjectPaths.RAW_DATA_DIR / "Screencast from 2026-07-10 17-15-03.mp4")
    DEFAULT_SAHI_VIDEO = str(ProjectPaths.RAW_DATA_DIR / "TownCentre_1min.mp4")
    OUTPUT_STANDARD_DIR = str(ProjectPaths.OUTPUTS_DIR / "count_standard")
    OUTPUT_SAHI_DIR = str(ProjectPaths.OUTPUTS_DIR / "count_sahi")


def setup_parser(default_video: str, default_model: str, default_output: str, description: str = "Person Counting Inference"):
    """
    Setup a standard argparse for all scripts in the project.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--video", type=str, default=default_video, help="Path to input video")
    parser.add_argument("--model", type=str, default=default_model, help="Path to model weights (.pt)")
    parser.add_argument("--output", type=str, default=default_output, help="Directory to save outputs")
    return parser
