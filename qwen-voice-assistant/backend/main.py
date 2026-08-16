"""
FastAPI server for Emotion-Aware Conversational AI Assistant.
Handles chat, TTS, user accounts, and emotion integration.
"""

import os
import uuid
import base64
from datetime import datetime, date
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_service import LLMService
from tts_service import TTSService
from emotion_handler import EmotionHandler
from user_service import UserService


llm_service: LLMService = None
tts_service: TTSService = None
emotion_handler: EmotionHandler = None
user_service: UserService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_service, tts_service, emotion_handler, user_service

    print("Loading services...")
    llm_service = LLMService()
    tts_service = TTSService()
    emotion_handler = EmotionHandler()
    user_service = UserService()
    print("Services loaded successfully")

    yield

    print("Shutting down services...")


app = FastAPI(
    title="Emotion-Aware Chat API",
    description="Chat API with emotion detection and voice synthesis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    birthday: str = Field(..., description="Birthday in YYYY-MM-DD format")


class UserResponse(BaseModel):
    user_id: str
    name: str
    age: int
    created_at: str


class ChatRequest(BaseModel):
    message: str
    user_id: str
    emotion: Optional[str] = "neutral"
    emotion_confidence: Optional[float] = 0.0


class ChatResponse(BaseModel):
    response: str
    detected_emotion: str
    adapted_tone: str
    audio_url: Optional[str] = None


class EmotionUpdate(BaseModel):
    user_id: str
    face_emotion: Optional[str] = None
    voice_emotion: Optional[str] = None
    confidence: float = 0.0
    face_confidence: Optional[float] = None
    voice_confidence: Optional[float] = None
    face_valence: Optional[float] = None
    face_arousal: Optional[float] = None
    voice_valence: Optional[float] = None
    voice_arousal: Optional[float] = None
    bridge_frame_jpeg: Optional[str] = None


bridge_frames = {}


class ExternalEmotionResponse(BaseModel):
    user_id: str
    fused_emotion: str
    confidence: float
    face_emotion: Optional[str] = None
    face_confidence: float = 0.0
    voice_emotion: Optional[str] = None
    voice_confidence: float = 0.0
    age_seconds: float


@app.get("/")
async def root():
    return FileResponse("../frontend/index.html")


@app.post("/api/users", response_model=UserResponse)
async def create_user(user_data: UserCreate):
    try:
        birthday = datetime.strptime(user_data.birthday, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    today = date.today()
    age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))

    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Invalid age")

    user = user_service.create_user(user_data.name, birthday, age)
    return user


@app.get("/api/users")
async def list_users():
    return user_service.list_users()


@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user = user_service.get_user(request.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    final_emotion = emotion_handler.combine_emotions(
        text_emotion=emotion_handler.detect_text_emotion(request.message),
        provided_emotion=request.emotion,
        confidence=request.emotion_confidence,
        user_id=request.user_id,
    )

    adapted_prompt = emotion_handler.get_adapted_prompt(final_emotion, user["age"])

    response_text, conversation = llm_service.chat(
        user_id=request.user_id,
        message=request.message,
        system_prompt=adapted_prompt,
    )

    audio_url = None
    if tts_service.is_available():
        try:
            speaker_name = emotion_handler.get_tts_speaker(final_emotion)
            tts_text = emotion_handler.add_tts_tone_hint(response_text, final_emotion)
            audio_path = tts_service.synthesize(tts_text, speaker=speaker_name)
            audio_url = f"/api/audio/{os.path.basename(audio_path)}"
        except Exception as e:
            print(f"TTS failed: {e}")

    return ChatResponse(
        response=response_text,
        detected_emotion=final_emotion,
        adapted_tone=emotion_handler.get_tone_description(final_emotion),
        audio_url=audio_url,
    )


@app.post("/api/emotion")
async def update_emotion(emotion_data: EmotionUpdate):
    """Receive emotion updates from Jetson edge device."""
    prev = emotion_handler.get_external_emotion(emotion_data.user_id)

    face_emotion = emotion_data.face_emotion if emotion_data.face_emotion is not None else (prev.face_emotion if prev else None)
    voice_emotion = emotion_data.voice_emotion if emotion_data.voice_emotion is not None else (prev.voice_emotion if prev else None)

    face_confidence = emotion_data.face_confidence
    if face_confidence is None:
        if emotion_data.face_emotion is not None and emotion_data.confidence > 0:
            face_confidence = emotion_data.confidence
        elif prev is not None:
            face_confidence = prev.face_confidence

    voice_confidence = emotion_data.voice_confidence
    if voice_confidence is None:
        if emotion_data.voice_emotion is not None and emotion_data.confidence > 0:
            voice_confidence = emotion_data.confidence
        elif prev is not None:
            voice_confidence = prev.voice_confidence

    face_valence = emotion_data.face_valence if emotion_data.face_valence is not None else (prev.face_valence if prev else None)
    face_arousal = emotion_data.face_arousal if emotion_data.face_arousal is not None else (prev.face_arousal if prev else None)
    voice_valence = emotion_data.voice_valence if emotion_data.voice_valence is not None else (prev.voice_valence if prev else None)
    voice_arousal = emotion_data.voice_arousal if emotion_data.voice_arousal is not None else (prev.voice_arousal if prev else None)

    merged_confidence = emotion_data.confidence
    if merged_confidence <= 0 and prev is not None:
        merged_confidence = prev.confidence

    emotion_handler.update_external_emotion(
        user_id=emotion_data.user_id,
        face_emotion=face_emotion,
        voice_emotion=voice_emotion,
        confidence=merged_confidence,
        face_confidence=face_confidence,
        voice_confidence=voice_confidence,
        face_valence=face_valence,
        face_arousal=face_arousal,
        voice_valence=voice_valence,
        voice_arousal=voice_arousal,
    )

    if emotion_data.bridge_frame_jpeg:
        try:
            image_bytes = base64.b64decode(emotion_data.bridge_frame_jpeg, validate=True)
            bridge_frames[emotion_data.user_id] = {
                "image": image_bytes,
                "timestamp": datetime.now(),
            }
        except Exception:
            pass

    return {"status": "ok"}


@app.get("/api/bridge-frame/{user_id}")
async def get_bridge_frame(user_id: str):
    frame = bridge_frames.get(user_id)
    if not frame:
        raise HTTPException(status_code=404, detail="No bridge frame available")

    age_seconds = (datetime.now() - frame["timestamp"]).total_seconds()
    if age_seconds > 5.0:
        bridge_frames.pop(user_id, None)
        raise HTTPException(status_code=404, detail="Bridge frame is stale")

    return Response(content=frame["image"], media_type="image/jpeg")


@app.get("/api/emotion/{user_id}", response_model=ExternalEmotionResponse)
async def get_external_emotion(user_id: str):
    external = emotion_handler.get_external_emotion(user_id)
    if not external:
        raise HTTPException(status_code=404, detail="No recent emotion data")

    age_seconds = (datetime.now() - external.timestamp).total_seconds()
    return ExternalEmotionResponse(
        user_id=user_id,
        fused_emotion=external.fused_emotion,
        confidence=external.confidence,
        face_emotion=external.face_emotion,
        face_confidence=external.face_confidence,
        voice_emotion=external.voice_emotion,
        voice_confidence=external.voice_confidence,
        age_seconds=age_seconds,
    )


@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    audio_path = os.path.join(tts_service.output_dir, filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(audio_path, media_type="audio/wav")


@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    await websocket.accept()

    user = user_service.get_user(user_id)
    if not user:
        await websocket.close(code=4004, reason="User not found")
        return

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            emotion = data.get("emotion", "neutral")
            confidence = data.get("confidence", 0.0)

            final_emotion = emotion_handler.combine_emotions(
                text_emotion=emotion_handler.detect_text_emotion(message),
                provided_emotion=emotion,
                confidence=confidence,
                user_id=user_id,
            )

            adapted_prompt = emotion_handler.get_adapted_prompt(final_emotion, user["age"])

            response_text, _ = llm_service.chat(
                user_id=user_id,
                message=message,
                system_prompt=adapted_prompt,
            )

            await websocket.send_json({
                "response": response_text,
                "emotion": final_emotion,
                "tone": emotion_handler.get_tone_description(final_emotion),
            })

    except WebSocketDisconnect:
        print(f"User {user_id} disconnected")


app.mount("/static", StaticFiles(directory="../frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
