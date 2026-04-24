"""
Utility functions for video processing and helper operations
"""

import os
import cv2
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handle video processing and frame extraction"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.fps = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self._initialize()
    
    def _initialize(self):
        """Initialize video capture"""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video loaded: {self.total_frames} frames @ {self.fps} FPS")
        except Exception as e:
            logger.error(f"Error initializing video: {e}")
    
    def get_frame(self, frame_idx: int) -> Optional[Tuple]:
        """Get a specific frame from the video"""
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            logger.error(f"Error getting frame {frame_idx}: {e}")
            return None
    
    def get_video_info(self) -> Dict:
        """Get video metadata"""
        return {
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height,
            'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0,
            'path': self.video_path
        }
    
    def extract_frames_batch(self, frame_indices: List[int]) -> Dict[int, any]:
        """Extract multiple frames"""
        frames = {}
        for idx in frame_indices:
            frame = self.get_frame(idx)
            if frame is not None:
                frames[idx] = frame
        return frames
    
    def close(self):
        """Release video resources"""
        if self.cap:
            self.cap.release()
    
    def __del__(self):
        self.close()


class EventAggregator:
    """Aggregate DSG events into meaningful sequences"""
    
    def __init__(self, max_events: int = 100):
        self.events = []
        self.max_events = max_events
    
    def add_event(self, event: Dict):
        """Add an event to the log"""
        event['timestamp'] = datetime.now().isoformat()
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_recent_events(self, count: int = 10) -> List[Dict]:
        """Get the N most recent events"""
        return self.events[-count:]
    
    def get_events_by_type(self, event_type: str) -> List[Dict]:
        """Get all events of a specific type"""
        return [e for e in self.events if e.get('event') == event_type]
    
    def get_event_stats(self) -> Dict:
        """Get statistics about events"""
        event_types = {}
        for event in self.events:
            event_type = event.get('event', 'Unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            'total_events': len(self.events),
            'unique_types': len(event_types),
            'event_breakdown': event_types
        }
    
    def clear(self):
        """Clear event history"""
        self.events = []


class MatchContextManager:
    """Manage cricket match context for commentary"""
    
    def __init__(self):
        self.match_state = {
            'runs': 0,
            'wickets': 0,
            'overs': 0,
            'balls': 0,
            'striker': 'Player 1',
            'bowler': 'Bowler 1',
            'phase': 'Powerplay',
            'run_rate': 0.0,
            'required_rate': 0.0
        }
        self.delivery_log = []
        self.event_context = {}
    
    def update_match_state(self, **kwargs):
        """Update match state"""
        for key, value in kwargs.items():
            if key in self.match_state:
                self.match_state[key] = value
        
        self._calculate_rates()
    
    def _calculate_rates(self):
        """Calculate run rate and other metrics"""
        overs = self.match_state['overs']
        balls = self.match_state['balls']
        runs = self.match_state['runs']
        
        total_balls = overs * 6 + balls
        if total_balls > 0:
            self.match_state['run_rate'] = (runs * 6) / total_balls
    
    def log_delivery(self, delivery_info: Dict):
        """Log a delivery for context"""
        delivery_info['timestamp'] = datetime.now().isoformat()
        self.delivery_log.append(delivery_info)
    
    def get_context_summary(self) -> str:
        """Get a summary of the match context for LLM"""
        ms = self.match_state
        return f"""
Match Status:
- Score: {ms['runs']}/{ms['wickets']} ({ms['overs']}.{ms['balls']} overs)
- Run Rate: {ms['run_rate']:.2f}
- Striker: {ms['striker']}
- Bowler: {ms['bowler']}
- Phase: {ms['phase']}
        """.strip()
    
    def reset(self):
        """Reset match context"""
        self.match_state = {
            'runs': 0,
            'wickets': 0,
            'overs': 0,
            'balls': 0,
            'striker': 'Player 1',
            'bowler': 'Bowler 1',
            'phase': 'Powerplay',
            'run_rate': 0.0,
            'required_rate': 0.0
        }
        self.delivery_log = []
        self.event_context = {}


class CommentaryBuilder:
    """Build coherent commentary from events"""
    
    def __init__(self):
        self.previous_comments = []
        self.max_history = 5
    
    def build_context_prompt(self, current_event: str, recent_events: List[Dict]) -> str:
        """Build a context-aware prompt for the LLM"""
        
        context = "Recent play:\n"
        for event in recent_events[-3:]:  # Last 3 events
            context += f"- {event.get('event')}\n"
        
        context += f"\nCurrent event: {current_event}"
        
        return context
    
    def add_commentary(self, commentary: str, event: str):
        """Add commentary to history"""
        self.previous_comments.append({
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'commentary': commentary
        })
        
        # Keep only recent commentary
        if len(self.previous_comments) > self.max_history:
            self.previous_comments = self.previous_comments[-self.max_history:]
    
    def get_commentary_context(self) -> str:
        """Get previous commentary for continuity"""
        if not self.previous_comments:
            return "No previous commentary."
        
        recent = self.previous_comments[-1]
        return f"Last said: '{recent['commentary']}' about {recent['event']}"


class FrameAnalyzer:
    """Analyze individual video frames for objects and metrics"""
    
    @staticmethod
    def extract_regions_of_interest(frame, detections: List[Dict]) -> Dict:
        """Extract ROI from frame based on detections"""
        rois = {}
        
        for detection in detections:
            obj_type = detection.get('type', 'unknown')
            bbox = detection.get('box', [0, 0, 0, 0])
            
            x1, y1, x2, y2 = [int(v) for v in bbox]
            roi = frame[y1:y2, x1:x2]
            
            rois[obj_type] = {
                'roi': roi,
                'bbox': bbox,
                'confidence': detection.get('confidence', 0.0)
            }
        
        return rois
    
    @staticmethod
    def calculate_object_metrics(detections: List[Dict]) -> Dict:
        """Calculate metrics from detected objects"""
        metrics = {}
        
        for detection in detections:
            obj_type = detection.get('type', 'unknown')
            velocity = detection.get('velocity_kph', 0)
            
            if obj_type not in metrics:
                metrics[obj_type] = {
                    'count': 0,
                    'avg_velocity': 0,
                    'max_velocity': 0
                }
            
            metrics[obj_type]['count'] += 1
            metrics[obj_type]['max_velocity'] = max(
                metrics[obj_type]['max_velocity'],
                velocity
            )
        
        return metrics


class ConfigurationManager:
    """Manage application configuration"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = {}
        if config_file:
            self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value


# Utility functions

def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS format"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def get_asset_files(asset_dir: str, extensions: set = None) -> List[str]:
    """Get files from assets directory with specified extensions"""
    if extensions is None:
        extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
    
    files = []
    try:
        for file in os.listdir(asset_dir):
            if os.path.splitext(file)[1].lower() in extensions:
                files.append(file)
    except Exception as e:
        logger.error(f"Error reading assets: {e}")
    
    return sorted(files)


def ensure_directory_exists(path: str):
    """Ensure directory exists, create if not"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory: {e}")


if __name__ == "__main__":
    # Example usage
    print("Video Processor Example:")
    # processor = VideoProcessor("path/to/video.mp4")
    # info = processor.get_video_info()
    # print(info)
    
    print("Event Aggregator Example:")
    agg = EventAggregator()
    agg.add_event({'event': 'SHOT_PLAYED', 'actor': 'Batsman'})
    agg.add_event({'event': 'CATCH', 'actor': 'Fielder'})
    print(agg.get_event_stats())
    
    print("Match Context Manager Example:")
    manager = MatchContextManager()
    manager.update_match_state(runs=50, wickets=2, overs=8, balls=3)
    print(manager.get_context_summary())
