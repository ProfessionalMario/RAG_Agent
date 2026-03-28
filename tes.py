import cv2
import easyocr

# 1. Initialize EasyOCR (runs on CPU if gpu=False)
reader = easyocr.Reader(['en'], gpu=False)

# 2. Load the video
video_path = (r"C:\Users\manik\Videos\Animal_Instinct_Proposal.mp4")
cap = cv2.VideoCapture(video_path)

# Get some video info
fps = cap.get(cv2.CAP_PROP_FPS)
interval = 2  # Extract text every 2 seconds
frame_skip = int(fps * interval)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process only the 'interval' frames
    if frame_count % frame_skip == 0:
        # 3. Focus on the bottom (where captions are)
        # Height: from 80% to 100%, Width: all
        h, w, _ = frame.shape
        caption_region = frame[int(h*0.8):h, :]
        
        # 4. Extract Text
        timestamp = frame_count / fps
        results = reader.readtext(caption_region, detail=0, paragraph=True)
        
        if results:
            print(f"[{timestamp:.2f}s] {results[0]}")
            
    frame_count += 1

cap.release()
print("Done!")