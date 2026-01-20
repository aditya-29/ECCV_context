"""
Video Processing Utilities
Provides modular video processing capabilities including:
- Uniform sampling at configurable FPS
- Key frame detection
- Adaptive sampling based on scene changes
"""

import cv2
import numpy as np
import base64
from pathlib import Path
from typing import List, Optional, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Video frame sampling strategies."""
    UNIFORM = "uniform"  # Sample at fixed FPS
    KEY_FRAME = "key_frame"  # Detect and extract key frames
    ADAPTIVE = "adaptive"  # Adaptive sampling based on scene changes


class VideoProcessor:
    """
    Modular video processor with multiple sampling strategies.
    
    Supports:
    - Uniform sampling at configurable FPS
    - Key frame detection
    - Adaptive sampling based on scene changes
    """
    
    def __init__(
        self,
        sampling_strategy: Union[SamplingStrategy, str] = SamplingStrategy.UNIFORM,
        target_fps: float = 1.0,
        max_frames: Optional[int] = None,
        scene_threshold: float = 30.0,
        resize_dimensions: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the video processor.
        
        Args:
            sampling_strategy: Strategy for sampling frames
            target_fps: Target frames per second for uniform sampling
            max_frames: Maximum number of frames to extract (None = unlimited)
            scene_threshold: Threshold for scene change detection (0-100)
            resize_dimensions: Optional (width, height) to resize frames
        """
        if isinstance(sampling_strategy, str):
            sampling_strategy = SamplingStrategy(sampling_strategy)
        
        self.sampling_strategy = sampling_strategy
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.scene_threshold = scene_threshold
        self.resize_dimensions = resize_dimensions
        
        logger.info(
            f"VideoProcessor initialized: strategy={sampling_strategy.value}, "
            f"target_fps={target_fps}, max_frames={max_frames}"
        )
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_format: str = "base64"
    ) -> List[Union[str, np.ndarray]]:
        """
        Process video and extract frames based on sampling strategy.
        
        Args:
            video_path: Path to the video file
            output_format: Output format - "base64" or "numpy"
            
        Returns:
            List of frames (base64 strings or numpy arrays)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            
            logger.info(
                f"Video properties: fps={video_fps:.2f}, "
                f"total_frames={total_frames}, duration={duration:.2f}s"
            )
            
            # Extract frames based on strategy
            if self.sampling_strategy == SamplingStrategy.UNIFORM:
                frames = self._uniform_sampling(cap, video_fps, total_frames)
            elif self.sampling_strategy == SamplingStrategy.KEY_FRAME:
                frames = self._key_frame_detection(cap)
            elif self.sampling_strategy == SamplingStrategy.ADAPTIVE:
                frames = self._adaptive_sampling(cap)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
            logger.info(f"Extracted {len(frames)} frames")
            
            # Convert to requested format
            if output_format == "base64":
                return [self._frame_to_base64(frame) for frame in frames]
            elif output_format == "numpy":
                return frames
            else:
                raise ValueError(f"Unknown output format: {output_format}")
        
        finally:
            cap.release()
    
    def _uniform_sampling(
        self,
        cap: cv2.VideoCapture,
        video_fps: float,
        total_frames: int
    ) -> List[np.ndarray]:
        """
        Extract frames at uniform intervals based on target FPS.
        
        Args:
            cap: OpenCV VideoCapture object
            video_fps: Original video FPS
            total_frames: Total frames in video
            
        Returns:
            List of extracted frames
        """
        frames = []
        frame_interval = max(1, int(video_fps / self.target_fps))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frame at intervals
            if frame_idx % frame_interval == 0:
                processed_frame = self._process_frame(frame)
                frames.append(processed_frame)
                
                # Check max frames limit
                if self.max_frames and len(frames) >= self.max_frames:
                    logger.info(f"Reached max frames limit: {self.max_frames}")
                    break
            
            frame_idx += 1
        
        return frames
    
    def _key_frame_detection(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """
        Detect and extract key frames using difference-based detection.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            List of key frames
        """
        frames = []
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # First frame is always a key frame
            if prev_frame is None:
                processed_frame = self._process_frame(frame)
                frames.append(processed_frame)
                prev_frame = gray
                continue
            
            # Calculate difference from previous frame
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(diff)
            
            # If difference exceeds threshold, consider it a key frame
            if mean_diff > self.scene_threshold:
                processed_frame = self._process_frame(frame)
                frames.append(processed_frame)
                prev_frame = gray
                
                # Check max frames limit
                if self.max_frames and len(frames) >= self.max_frames:
                    logger.info(f"Reached max frames limit: {self.max_frames}")
                    break
        
        return frames
    
    def _adaptive_sampling(self, cap: cv2.VideoCapture) -> List[np.ndarray]:
        """
        Adaptive sampling based on scene changes.
        Samples more densely during high-motion scenes.
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            List of adaptively sampled frames
        """
        frames = []
        prev_frame = None
        motion_history = []
        base_skip = 10  # Base frame skip
        
        frame_idx = 0
        frames_since_last = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # First frame
            if prev_frame is None:
                processed_frame = self._process_frame(frame)
                frames.append(processed_frame)
                prev_frame = gray
                frame_idx += 1
                continue
            
            # Calculate motion
            diff = cv2.absdiff(gray, prev_frame)
            motion = np.mean(diff)
            motion_history.append(motion)
            
            # Keep only recent history
            if len(motion_history) > 30:
                motion_history.pop(0)
            
            # Calculate adaptive skip based on recent motion
            avg_motion = np.mean(motion_history)
            if avg_motion > self.scene_threshold:
                adaptive_skip = max(3, base_skip // 3)  # Sample more frequently
            else:
                adaptive_skip = base_skip  # Sample less frequently
            
            frames_since_last += 1
            
            # Sample frame if skip threshold met
            if frames_since_last >= adaptive_skip:
                processed_frame = self._process_frame(frame)
                frames.append(processed_frame)
                prev_frame = gray
                frames_since_last = 0
                
                # Check max frames limit
                if self.max_frames and len(frames) >= self.max_frames:
                    logger.info(f"Reached max frames limit: {self.max_frames}")
                    break
            
            frame_idx += 1
        
        return frames
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame (resize if needed).
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        if self.resize_dimensions:
            width, height = self.resize_dimensions
            frame = cv2.resize(frame, (width, height))
        return frame
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert frame to base64 encoded JPEG string.
        
        Args:
            frame: Frame as numpy array
            
        Returns:
            Base64 encoded string
        """
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        # Convert to base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return base64_str
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            return {
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration": duration,
                "file_size": video_path.stat().st_size,
            }
        finally:
            cap.release()


def process_video_uniform(
    video_path: Union[str, Path],
    fps: float = 1.0,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> List[str]:
    """
    Convenience function for uniform sampling.
    
    Args:
        video_path: Path to video file
        fps: Target frames per second
        max_frames: Maximum frames to extract
        resize: Optional (width, height) for resizing
        
    Returns:
        List of base64 encoded frames
    """
    processor = VideoProcessor(
        sampling_strategy=SamplingStrategy.UNIFORM,
        target_fps=fps,
        max_frames=max_frames,
        resize_dimensions=resize
    )
    return processor.process_video(video_path, output_format="base64")


def process_video_keyframes(
    video_path: Union[str, Path],
    threshold: float = 30.0,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> List[str]:
    """
    Convenience function for key frame detection.
    
    Args:
        video_path: Path to video file
        threshold: Scene change threshold (0-100)
        max_frames: Maximum frames to extract
        resize: Optional (width, height) for resizing
        
    Returns:
        List of base64 encoded frames
    """
    processor = VideoProcessor(
        sampling_strategy=SamplingStrategy.KEY_FRAME,
        scene_threshold=threshold,
        max_frames=max_frames,
        resize_dimensions=resize
    )
    return processor.process_video(video_path, output_format="base64")


def process_video_adaptive(
    video_path: Union[str, Path],
    threshold: float = 30.0,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> List[str]:
    """
    Convenience function for adaptive sampling.
    
    Args:
        video_path: Path to video file
        threshold: Motion detection threshold
        max_frames: Maximum frames to extract
        resize: Optional (width, height) for resizing
        
    Returns:
        List of base64 encoded frames
    """
    processor = VideoProcessor(
        sampling_strategy=SamplingStrategy.ADAPTIVE,
        scene_threshold=threshold,
        max_frames=max_frames,
        resize_dimensions=resize
    )
    return processor.process_video(video_path, output_format="base64")