import cv2
import math
import tkinter as tk
from tkinter import filedialog
import os
import sys
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import Counter

class UnifiedCricketAnalyzer:
    def __init__(self):
        # Model paths
        self.model_paths = {
            'main': 'D:/Downloads/Cricket/YOLO/runs/detect/train/weights/best.pt',
            'shot': 'D:/Downloads/Cricket/Yolo_2/runs/detect/train/weights/best.pt',
            'position': 'D:/Downloads/Cricket/Yolo_3/runs/detect/train/weights/best.pt'
        }
        
        # Load models
        self.models = {}
        for key, path in self.model_paths.items():
            try:
                self.models[key] = YOLO(path)
                print(f"✅ Loaded {key} model")
            except Exception as e:
                print(f"❌ Failed to load {key} model: {e}")
        
        # Class mappings
        self.class_names = {
            'shot': {
                0: "Drive",
                1: "Defensive", 
                2: "Aggressive",
                3: "Leave"
            },
            'position': {
                0: "Left Hand",
                1: "Right Hand"
            }
        }
        
        # Colors for different detections
        self.colors = {
            'ball': (0, 255, 0),      # Green
            'bat': (255, 0, 0),       # Blue
            'batsman': (255, 165, 0), # Orange
            'bowler': (0, 0, 255),    # Red
            'fielder': (255, 255, 0), # Cyan
            'stumps': (255, 0, 255),  # Magenta
            'umpire': (0, 255, 255),  # Yellow
            'wk': (128, 128, 128)     # Gray
        }
        
        # Tracking variables
        self.ball_path = []
        self.bat_path = []
        self.bowler_path = []
        self.prev_ball_center = None
        self.max_ball_speed = 0
        self.frame_count = 0
        
        # Shot analysis
        self.shot_counter = Counter()
        self.current_shot = None
        self.current_shot_conf = 0
        
        # Counterfactual analysis
        self.counterfactuals = {
            'fast': [], 'faster': [], 'slow': [], 'slower': []
        }
        self.speed_factors = {
            'fast': 1.10, 'faster': 1.20, 'slow': 0.90, 'slower': 0.80
        }
        
        # Player positioning
        self.view_type = 'front'  # Default to front view
        
    def select_video(self):
        """Select video file using file dialog"""
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="🎬 Select Cricket Video for Analysis",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        return video_path
    
    def setup_output(self, video_path):
        """Setup output directory and video writer"""
        # Create output directory
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_folder = os.path.join(desktop_path, "Cricket_Analysis")
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate output filename
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{video_name}_unified_analysis_{timestamp}.mp4"
        output_path = os.path.join(output_folder, output_filename)
        
        return output_path, output_folder
    
    def calculate_ball_speed(self, current_center, time_per_frame):
        """Calculate ball speed between frames"""
        if self.prev_ball_center is not None:
            distance = math.hypot(
                current_center[0] - self.prev_ball_center[0],
                current_center[1] - self.prev_ball_center[1]
            )
            speed = distance / time_per_frame
            self.max_ball_speed = max(self.max_ball_speed, speed)
            return speed
        return 0
    
    def update_counterfactuals(self, current_center):
        """Update counterfactual ball trajectories"""
        if len(self.ball_path) >= 2:
            prev_real = self.ball_path[-2]
            dx = current_center[0] - prev_real[0]
            dy = current_center[1] - prev_real[1]
            
            for key, factor in self.speed_factors.items():
                if len(self.counterfactuals[key]) == 0:
                    self.counterfactuals[key].append(prev_real)
                last_point = self.counterfactuals[key][-1]
                new_point = (
                    int(last_point[0] + dx * factor),
                    int(last_point[1] + dy * factor)
                )
                self.counterfactuals[key].append(new_point)
    
    def draw_detections(self, frame, results, model_type='main'):
        """Draw bounding boxes for detected objects"""
        if not results or not results[0].boxes:
            return
        
        boxes = results[0].boxes.xywh.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confidences):
            x_center, y_center, w, h = box
            cls = int(cls)
            conf = float(conf)
            
            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
            
            # Determine color and label based on class
            if model_type == 'main':
                if cls == 0:  # Bat
                    color = self.colors['bat']
                    label = f"Bat ({conf:.2f})"
                    current_center = (int(x_center), int(y_center))
                    self.bat_path.append(current_center)
                    
                elif cls == 1:  # Ball
                    color = self.colors['ball']
                    label = f"Ball ({conf:.2f})"
                    current_center = (int(x_center), int(y_center))
                    self.ball_path.append(current_center)
                    
                    # Calculate speed
                    time_per_frame = 1 / self.fps
                    speed = self.calculate_ball_speed(current_center, time_per_frame)
                    
                    # Update counterfactuals
                    self.update_counterfactuals(current_center)
                    
                    # Update previous position
                    self.prev_ball_center = current_center
                    
                elif cls == 2:  # Batsman
                    color = self.colors['batsman']
                    label = f"Batsman ({conf:.2f})"
                    
                elif cls == 3:  # Bowler
                    color = self.colors['bowler']
                    label = f"Bowler ({conf:.2f})"
                    bowler_center = (int(x_center), int(y_center))
                    self.bowler_path.append(bowler_center)
                    
                elif cls == 4:  # Fielder
                    color = self.colors['fielder']
                    label = f"Fielder ({conf:.2f})"
                    
                elif cls == 5:  # Stumps
                    color = self.colors['stumps']
                    label = f"Stumps ({conf:.2f})"
                    
                elif cls == 6:  # Umpire
                    color = self.colors['umpire']
                    label = f"Umpire ({conf:.2f})"
                    
                elif cls == 7:  # Wicket Keeper
                    color = self.colors['wk']
                    label = f"WK ({conf:.2f})"
                    
                else:
                    color = (128, 128, 128)
                    label = f"Class {cls} ({conf:.2f})"
            
            elif model_type == 'shot':
                if cls in self.class_names['shot']:
                    color = (0, 255, 255)  # Yellow for shots
                    label = f"{self.class_names['shot'][cls]} ({conf:.2f})"
                    self.shot_counter[cls] += 1
                    self.current_shot = cls
                    self.current_shot_conf = conf
                else:
                    continue
                    
            elif model_type == 'position':
                if cls in self.class_names['position']:
                    color = (255, 0, 255)  # Magenta for positioning
                    label = f"{self.class_names['position'][cls]} ({conf:.2f})"
                else:
                    continue
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_paths(self, frame):
        """Draw tracking paths for ball, bat, and bowler"""
        # Draw ball path
        if len(self.ball_path) > 2:
            pts = np.array(self.ball_path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, self.colors['ball'], 3)
        
        # Draw bat path
        if len(self.bat_path) > 2:
            pts = np.array(self.bat_path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, self.colors['bat'], 2)
        
        # Draw bowler path
        if len(self.bowler_path) > 2:
            pts = np.array(self.bowler_path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, self.colors['bowler'], 2)
        
        # Draw counterfactual paths
        counterfactual_colors = {
            'fast': (0, 0, 255),      # Red
            'faster': (255, 0, 0),    # Blue
            'slow': (0, 255, 255),    # Yellow
            'slower': (255, 0, 255)   # Magenta
        }
        
        for key, path in self.counterfactuals.items():
            if len(path) > 2:
                pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, counterfactual_colors[key], 2)
    
    def draw_analytics(self, frame):
        """Draw analytics information on frame"""
        height, width = frame.shape[:2]
        
        # Speed information
        if self.max_ball_speed > 0:
            cv2.putText(frame, f"Max Ball Speed: {self.max_ball_speed:.1f} px/sec", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Shot analysis
        if self.current_shot is not None:
            shot_name = self.class_names['shot'].get(self.current_shot, 'Unknown')
            cv2.putText(frame, f"Current Shot: {shot_name} ({self.current_shot_conf:.2f})", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", 
                   (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend for counterfactuals
        legend_y = height - 150
        legends = [
            ("Real Path (Green)", self.colors['ball']),
            ("Fast (Red)", (0, 0, 255)),
            ("Faster (Blue)", (255, 0, 0)),
            ("Slow (Yellow)", (0, 255, 255)),
            ("Slower (Pink)", (255, 0, 255))
        ]
        
        for text, color in legends:
            cv2.putText(frame, text, (20, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            legend_y += 25
    
    def draw_shot_chart(self, frame):
        """Draw shot distribution chart"""
        if not self.shot_counter:
            return
        
        height, width = frame.shape[:2]
        chart_x = width - 200
        chart_y = 80
        bar_width = 25
        max_bar_height = 100
        spacing = 50
        
        total = sum(self.shot_counter.values())
        if total == 0:
            return
        
        # Draw chart background
        cv2.rectangle(frame, (chart_x - 10, chart_y - 10), 
                     (chart_x + 150, chart_y + max_bar_height + 60), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (chart_x - 10, chart_y - 10), 
                     (chart_x + 150, chart_y + max_bar_height + 60), 
                     (255, 255, 255), 2)
        
        # Draw bars
        for i, (cls_id, count) in enumerate(self.shot_counter.items()):
            if cls_id in self.class_names['shot']:
                percentage = (count / total) * 100
                bar_height = int((percentage / 100) * max_bar_height)
                
                x1 = chart_x + i * spacing
                y1 = chart_y + max_bar_height
                x2 = x1 + bar_width
                y2 = y1 - bar_height
                
                # Bar color
                color = (0, 255, 255)  # Yellow
                
                # Draw bar
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                
                # Draw label
                label = self.class_names['shot'][cls_id]
                cv2.putText(frame, label, (x1 - 5, y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"{percentage:.1f}%", (x1, y2 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    
    def process_video(self, video_path):
        """Main video processing function"""
        print(f"🎬 Starting unified cricket analysis...")
        print(f"📁 Input video: {os.path.basename(video_path)}")
        
        # Setup video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error opening video file")
            return
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = self.fps if self.fps > 0 else 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video properties: {width}x{height} @ {self.fps:.2f} FPS")
        
        # Setup output
        output_path, output_folder = self.setup_output(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        print(f"📤 Output will be saved to: {output_path}")
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run all models on the frame
            results_main = self.models['main'].predict(source=frame, conf=0.5, save=False, verbose=False)
            results_shot = self.models['shot'].predict(source=frame, conf=0.5, save=False, verbose=False)
            results_position = self.models['position'].predict(source=frame, conf=0.5, save=False, verbose=False)
            
            # Draw detections from all models
            self.draw_detections(frame, results_main, 'main')
            self.draw_detections(frame, results_shot, 'shot')
            self.draw_detections(frame, results_position, 'position')
            
            # Draw paths and analytics
            self.draw_paths(frame)
            self.draw_analytics(frame)
            self.draw_shot_chart(frame)
            
            # Write frame to output video
            out.write(frame)
            
            # Display frame
            cv2.imshow('🏏 Unified Cricket Analysis', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("⏹️ Processing stopped by user")
                break
            elif key == ord('p'):
                print("⏸️ Paused - Press any key to continue")
                cv2.waitKey(0)
            
            self.frame_count += 1
            
            # Progress indicator
            if self.frame_count % 30 == 0:
                print(f"📈 Processed {self.frame_count} frames...")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Generate analysis report
        self.generate_report(output_folder, video_path)
        
        print(f"✅ Analysis complete!")
        print(f"📊 Total frames processed: {self.frame_count}")
        print(f"🎥 Output video: {output_path}")
    
    def generate_report(self, output_folder, video_path):
        """Generate analysis report"""
        report_path = os.path.join(output_folder, "analysis_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("🏏 CRICKET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input Video: {os.path.basename(video_path)}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {self.frame_count}\n")
            f.write(f"Video Duration: {self.frame_count / self.fps:.2f} seconds\n\n")
            
            f.write("📊 DETECTION STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Ball detections: {len(self.ball_path)}\n")
            f.write(f"Bat detections: {len(self.bat_path)}\n")
            f.write(f"Bowler detections: {len(self.bowler_path)}\n")
            f.write(f"Maximum ball speed: {self.max_ball_speed:.1f} px/sec\n\n")
            
            f.write("🏏 SHOT ANALYSIS\n")
            f.write("-" * 30 + "\n")
            total_shots = sum(self.shot_counter.values())
            if total_shots > 0:
                for cls_id, count in self.shot_counter.items():
                    if cls_id in self.class_names['shot']:
                        percentage = (count / total_shots) * 100
                        f.write(f"{self.class_names['shot'][cls_id]}: {count} ({percentage:.1f}%)\n")
            else:
                f.write("No shots detected\n")
            
            f.write(f"\nReport saved to: {report_path}\n")
        
        print(f"📋 Analysis report saved to: {report_path}")

def main():
    """Main function"""
    print("🏏 Welcome to Unified Cricket Analyzer!")
    print("=" * 50)
    
    # Create analyzer instance
    analyzer = UnifiedCricketAnalyzer()
    
    # Select video
    video_path = analyzer.select_video()
    if not video_path:
        print("❌ No video selected. Exiting...")
        return
    
    # Process video
    try:
        analyzer.process_video(video_path)
    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    
    print("👋 Thank you for using Unified Cricket Analyzer!")

if __name__ == "__main__":
    main() 