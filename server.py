# ... [imports remain the same] ...

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("models")
# MUST match training script
CLASS_NAMES = ["Mask", "No Mask"]

# ... [PREPROCESS and load_models() remain the same] ...

@socketio.on('video_frame')
def handle_frame(data):
    try:
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if frame is None: return

        h, w = frame.shape[:2]
        face = frame[int(h*0.15):int(h*0.85), int(w*0.25):int(w*0.75)]
        
        tensor = PREPROCESS(face).unsqueeze(0).to(DEVICE)
        ensemble_probs = torch.zeros(2, device=DEVICE)
        details = {}
        
        with torch.no_grad():
            for name, model in MODELS.items():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                ensemble_probs += probs
                conf, cls = torch.max(probs, dim=0)
                details[name] = {"label": CLASS_NAMES[cls.item()], "confidence": float(conf.item())}
                
        ensemble_probs /= len(MODELS)
        final_conf, final_cls = torch.max(ensemble_probs, dim=0)
        label = CLASS_NAMES[final_cls.item()]
        confidence = float(final_conf.item())
        
        # 🟢 Green for Mask, 🔴 Red for No Mask
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        cv2.rectangle(frame, (int(w*0.25), int(h*0.15)), (int(w*0.75), int(h*0.85)), color, 2)
        cv2.putText(frame, f"{label} {confidence:.0%}", (int(w*0.25), int(h*0.15)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = "image/jpeg;base64," + base64.b64encode(buffer).decode()
        
        emit('prediction_result', {
            'frame': frame_b64,
            'label': label,
            'confidence': round(confidence, 3),
            'details': details
        })
    except Exception as e:
        logging.error(f"Frame error: {e}")
        emit('error', {'msg': str(e)})

# ... [rest of server.py remains the same] ...