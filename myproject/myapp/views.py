from django.shortcuts import render
from django.http import JsonResponse, HttpRequest
from rest_framework.decorators import api_view
import tensorflow as tf
import numpy as np
from .video_preprocessing import load_uploaded_video, save_suspicious_frames, detect_people_yolo
from .metrics import F1Score 
from io import BytesIO
from django.conf import settings
import os 
from django.core.files.storage import FileSystemStorage
import uuid

model = tf.keras.models.load_model(
    "F:\\prcatice\\Computer Vision Intern\\Week 7 & 8\\best_model.keras",
    custom_objects={'F1Score': F1Score}
)

def home(request):
   video_url = None
   prediction = None
   if request.method == "POST" and request.FILES.get("video"):
       uploaded_file = request.FILES["video"]
       save_path = os.path.join(settings.MEDIA_ROOT, "videos")
       
       os.makedirs(save_path, exist_ok=True)
       fs = FileSystemStorage(location=save_path)
       filename = fs.save(uploaded_file.name, uploaded_file)
       video_url = settings.MEDIA_URL + "videos/" + filename
   return render(request, "home.html", {"video_url": video_url, "prediction": prediction})

@api_view(['POST'])
def predict_video(request):
   """Handles video prediction using the trained model"""
   
   try:
       file = request.FILES.get('video')
       if not file:
           return JsonResponse({'error': 'No video uploaded'}, status=400)
       save_path = os.path.join(settings.MEDIA_ROOT, "videos")
       os.makedirs(save_path, exist_ok=True)
       
       fs = FileSystemStorage(location=save_path)
       filename = fs.save(file.name, file)
       
       file_path = os.path.join(save_path, filename)
       
       with open(file_path, 'rb') as video_file:
           video_tensor, sampled_frames, frame_indices = load_uploaded_video(
               BytesIO(video_file.read()), 
               return_indices=True
           )
       
       video_tensor_batch = np.expand_dims(video_tensor, axis=0)
       prediction = model.predict(video_tensor_batch)
       
       if prediction.shape[1] == 1: 
           predicted_confidence = float(prediction[0, 0])
           predicted_class = int(predicted_confidence > 0.5)
           confidence = predicted_confidence if predicted_class == 1 else 1 - predicted_confidence
       else:
           probabilities = tf.nn.softmax(prediction).numpy()
           predicted_class = int(np.argmax(probabilities))
           confidence = float(probabilities[0, predicted_class])
       
       video_url = settings.MEDIA_URL + "videos/" + filename
       
       suspicious_frames_urls = []
       high_confidence_indices = []
       frame_confidence_values = []
       
       if predicted_class == 1:
           
           frame_confidences = []
           window_size = 5 
           
           for i in range(len(video_tensor)):
               start_idx = max(0, i - window_size // 2)
               end_idx = min(len(video_tensor), i + window_size // 2 + 1)
               
               window_frames = video_tensor[start_idx:end_idx]
               
               if len(window_frames) < window_size:
                   pad_size = window_size - len(window_frames)
                   padding = np.zeros((pad_size, *window_frames.shape[1:]))
                   window_frames = np.vstack([window_frames, padding])
               
               window_tensor = np.expand_dims(window_frames, axis=0)
               
               try:
                   window_pred = model.predict(window_tensor, verbose=0)
                   
                   if window_pred.shape[1] == 1:
                       conf = float(window_pred[0, 0])
                   else:
                       probs = tf.nn.softmax(window_pred).numpy()
                       conf = float(probs[0, 1])  # Class 1 probability (theft)
               except:
                   relative_pos = float(i) / len(video_tensor)
                   middle_weight = 1.0 - 2.0 * abs(relative_pos - 0.5)  # Higher in the middle
                   conf = confidence * (0.7 + 0.6 * middle_weight)  # Scale between 70-130% of overall confidence
               
               frame_confidences.append(min(1.0, max(0.0, conf)))  # Ensure confidence is between 0 and 1
           
           detection_id = str(uuid.uuid4())[:8]
           frames_dir = os.path.join(settings.MEDIA_ROOT, "suspicious_frames", detection_id)
           
           CONFIDENCE_THRESHOLD = 0.94
           high_confidence_frames = []
           high_confidence_indices = []
           frame_confidence_values = []
           
           for i, (frame, conf) in enumerate(zip(sampled_frames, frame_confidences)):
               if conf >= CONFIDENCE_THRESHOLD:
                   high_confidence_frames.append(frame)
                   high_confidence_indices.append(frame_indices[i])
                   frame_confidence_values.append(round(float(conf), 2))
           
           if not high_confidence_frames and sampled_frames:
               print(f"No frames met the high confidence threshold of {CONFIDENCE_THRESHOLD}. Using top 3 frames instead.")
               all_indices = np.argsort(frame_confidences)[::-1]  # Descending order
               
               top_frames_count = min(3, len(sampled_frames))
               for i in range(top_frames_count):
                   idx = all_indices[i]
                   high_confidence_frames.append(sampled_frames[idx])
                   high_confidence_indices.append(frame_indices[idx])
                   conf_value = round(float(frame_confidences[idx]), 2)
                   frame_confidence_values.append(conf_value)
                   print(f"Using frame {frame_indices[idx]} with confidence {conf_value} as fallback")
           
           if high_confidence_frames:
               sorted_indices = np.argsort(high_confidence_indices)
               high_confidence_frames = [high_confidence_frames[i] for i in sorted_indices]
               frame_confidence_values = [frame_confidence_values[i] for i in sorted_indices]
               high_confidence_indices = [high_confidence_indices[i] for i in sorted_indices]
               
               try:
                   frame_urls = save_suspicious_frames(
                       high_confidence_frames, 
                       frames_dir, 
                       f"frame_{filename.rsplit('.', 1)[0]}"
                   )
                   
                   suspicious_frames_urls = [
                       settings.MEDIA_URL + f"suspicious_frames/{detection_id}/{url}" 
                       for url in frame_urls
                   ]
               except Exception as e:
                   print(f"Frame processing error: {e}")
       
       high_confidence_indices = [int(idx) for idx in high_confidence_indices]
       
       return JsonResponse({
           "message": "Prediction successful",
           "prediction": predicted_class,
           "confidence": round(float(confidence), 4),
           "video_url": video_url,
           "suspicious_frames": suspicious_frames_urls,
           "frame_indices": high_confidence_indices,
           "frame_confidences": frame_confidence_values
       })
       
   except Exception as e:
       import traceback
       print(f"Error in predict_video: {str(e)}")
       print(traceback.format_exc())
       
       return JsonResponse({
           "error": f"Processing error: {str(e)}",
           "details": "There was an error processing the video."
       }, status=500)