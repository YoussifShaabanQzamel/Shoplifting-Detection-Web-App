import cv2
import numpy as np
import tempfile
import os

# Constants
IMG_SIZE = (128, 128)  
NUM_FRAMES = 16        
CHANNELS = 3           

def load_uploaded_video(uploaded_file, num_frames=NUM_FRAMES, return_indices=False):
    """
    Processes an uploaded video file:
    - Saves it temporarily.
    - Extracts a fixed number of frames.
    - Resizes and normalizes frames.
    - Returns as a NumPy array of shape (NUM_FRAMES, 128, 128, 3).
    - Optionally returns the frame indices that were sampled.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read()) 
        temp_video_path = temp_video.name      

    cap = cv2.VideoCapture(temp_video_path)
    frames = []

    if not cap.isOpened():
        print("[ERROR] Could not open video file.")
        if return_indices:
            return np.zeros((num_frames, *IMG_SIZE, CHANNELS), dtype=np.float32), []
        return np.zeros((num_frames, *IMG_SIZE, CHANNELS), dtype=np.float32)  

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames >= num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)  
    else:
        frame_indices = np.arange(total_frames) 

    sampled_frames = []
    processed_frames = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  
        ret, frame = cap.read()
        if not ret:
            break
            
        sampled_frames.append(cv2.resize(frame, (640, 480)))
        
        frame = cv2.resize(frame, IMG_SIZE)          
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0   
        processed_frames.append(frame)

    cap.release()
    os.remove(temp_video_path)  

    while len(processed_frames) < num_frames:
        processed_frames.append(processed_frames[-1] if processed_frames else np.zeros((*IMG_SIZE, CHANNELS)))
        sampled_frames.append(sampled_frames[-1] if sampled_frames else np.zeros((480, 640, CHANNELS)))

    if return_indices:
        return np.array(processed_frames, dtype=np.float32), sampled_frames, list(frame_indices)
    return np.array(processed_frames, dtype=np.float32)

def save_suspicious_frames(frames, save_dir, filename_base="suspicious_frame"):
    """
    Save frames with bounding boxes that better track the thief's movement
    Implemented using standard OpenCV functions without legacy module
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    
    if not frames or len(frames) == 0:
        print("Warning: No frames provided to save_suspicious_frames")
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "No frames available", (100, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        frame_path = os.path.join(save_dir, f"{filename_base}_error.jpg")
        cv2.imwrite(frame_path, blank_frame)
        return [os.path.basename(frame_path)]
    
    initial_frame = frames[0].copy()
    initial_detections = []
    best_detection = None
    
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        boxes, weights = hog.detectMultiScale(
            initial_frame, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        frame_width = initial_frame.shape[1]
        
        for i, (box, weight) in enumerate(zip(boxes, weights)):
            x, y, w, h = box
            if x < frame_width * 0.6 and weight > 0.3:
                initial_detections.append((box, weight))
    except Exception as e:
        print(f"HOG detection error: {str(e)}")
    
    if not initial_detections:
        height, width = initial_frame.shape[:2]
        x = int(width * 0.1)
        y = int(height * 0.2)
        w = int(width * 0.4)
        h = int(height * 0.6)
        initial_detections.append(((x, y, w, h), 0.5))
    
    initial_detections.sort(key=lambda x: x[1], reverse=True)
    best_detection = initial_detections[0][0]
    
    prev_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    
    x, y, w, h = best_detection
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    point_step = 10
    prev_points = []
    for ptx in range(x, x + w, point_step):
        for pty in range(y, y + h, point_step):
            if ptx < prev_frame.shape[1] and pty < prev_frame.shape[0]:
                prev_points.append([ptx, pty])
    
    prev_points = np.array(prev_points, dtype=np.float32).reshape(-1, 1, 2)
    
    current_box = [x, y, w, h]
    
    for i, frame in enumerate(frames):
        display_frame = frame.copy()
        
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if len(prev_points) > 0:
            try:
                new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_frame, 
                    curr_frame_gray, 
                    prev_points, 
                    None,
                    winSize=(15, 15),
                    maxLevel=2
                )
                
                good_new = new_points[status == 1]
                good_old = prev_points[status == 1]
                
                if len(good_new) >= 4:
                    movements = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
                    if len(movements) > 0:
                        dx = np.median(movements[:, 0])
                        dy = np.median(movements[:, 1])
                        
                        current_box[0] += dx
                        current_box[1] += dy
                        
                        current_box[0] = max(0, min(frame.shape[1] - current_box[2], current_box[0]))
                        current_box[1] = max(0, min(frame.shape[0] - current_box[3], current_box[1]))
                        
                        new_points = []
                        for ptx in range(int(current_box[0]), int(current_box[0] + current_box[2]), point_step):
                            for pty in range(int(current_box[1]), int(current_box[1] + current_box[3]), point_step):
                                if ptx < frame.shape[1] and pty < frame.shape[0]:
                                    new_points.append([ptx, pty])
                        
                        if new_points:
                            prev_points = np.array(new_points, dtype=np.float32).reshape(-1, 1, 2)
                        
                if i % 3 == 0:
                    try:
                        hog = cv2.HOGDescriptor()
                        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                        
                        boxes, weights = hog.detectMultiScale(
                            frame, 
                            winStride=(8, 8),
                            padding=(4, 4),
                            scale=1.05
                        )
                        
                        if len(boxes) > 0:
                            frame_width = frame.shape[1]
                            valid_detections = []
                            
                            for box, weight in zip(boxes, weights):
                                x, y, w, h = box
                                if x < frame_width * 0.6 and weight > 0.3:
                                    valid_detections.append((box, weight))
                            
                            if valid_detections:
                                valid_detections.sort(key=lambda x: x[1], reverse=True)
                                best_box, _ = valid_detections[0]
                                
                                current_box = list(best_box)
                                
                                new_points = []
                                x, y, w, h = current_box
                                for ptx in range(int(x), int(x + w), point_step):
                                    for pty in range(int(y), int(y + h), point_step):
                                        if ptx < frame.shape[1] and pty < frame.shape[0]:
                                            new_points.append([ptx, pty])
                                
                                if new_points:
                                    prev_points = np.array(new_points, dtype=np.float32).reshape(-1, 1, 2)
                    except Exception as e:
                        print(f"Re-detection error: {str(e)}")
            except Exception as e:
                print(f"Tracking error: {str(e)}")
        
        x, y, w, h = [int(v) for v in current_box]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        label_text = "Theft Suspect"
        
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            display_frame, 
            (x, y - text_size[1] - 10), 
            (x + text_size[0], y), 
            (0, 0, 0), 
            -1
        )
        
        cv2.putText(
            display_frame, 
            label_text, 
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        cv2.putText(
            display_frame, 
            "Theft Detection", 
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 255), 
            2
        )
        
        prev_frame = curr_frame_gray.copy()
        
        frame_path = os.path.join(save_dir, f"{filename_base}_{i}.jpg")
        cv2.imwrite(frame_path, display_frame)
        paths.append(os.path.basename(frame_path))
    
    return paths

def detect_people_yolo(frames, save_dir, filename_base="suspicious_frame"):
    """
    Simplified detection that uses the improved tracking approach
    """
    return save_suspicious_frames(frames, save_dir, filename_base)



