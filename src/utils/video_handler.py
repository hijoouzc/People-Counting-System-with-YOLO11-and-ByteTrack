"""
Video Handler Module.
Encapsulates OpenCV video reading and writing functionalities.
"""
import cv2
import sys
import os

class VideoHandler:
    """Class to manage video input streams and output writers."""
    
    def __init__(self, input_path: str, output_path: str = None):
        """
        Initialize the VideoHandler.
        
        Args:
            input_path (str): Path to the input video file.
            output_path (str, optional): Path to save the output video.
        """
        self.input_path = input_path
        self.output_path = output_path
        
        if not os.path.exists(self.input_path):
            print(f"[ERROR] Video file not found: {self.input_path}")
            sys.exit(1)
            
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open video: {self.input_path}")
            sys.exit(1)
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.writer = None
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.writer = cv2.VideoWriter(
                self.output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                self.fps, 
                (self.width, self.height)
            )

    def get_frames(self):
        """
        Generator to yield frames from the video.
        
        Yields:
            tuple: (success (bool), frame)
        """
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            yield frame
            
    def write_frame(self, frame):
        """
        Write a frame to the output video if writer is initialized.
        
        Args:
            frame: The video frame to write.
        """
        if self.writer:
            self.writer.write(frame)
            
    def release(self):
        """Release resources."""
        self.cap.release()
        if self.writer:
            self.writer.release()
