"""
Emotion Handler for detecting and adapting to user emotions.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExternalEmotion:
    face_emotion: Optional[str]
    voice_emotion: Optional[str]
    face_confidence: float
    voice_confidence: float
    face_valence: Optional[float]
    face_arousal: Optional[float]
    voice_valence: Optional[float]
    voice_arousal: Optional[float]
    confidence: float
    fused_emotion: str
    timestamp: datetime


class EmotionHandler:
    EMOTIONS = [
        "happy", "sad", "angry", "surprised",
        "disgusted", "fearful", "anxious", "frustrated", "neutral"
    ]

    EMOTION_KEYWORDS = {
        "frustrated": ["angry", "mad", "frustrated", "annoyed", "irritated"],
        "sad": ["sad", "upset", "hurt", "depressed", "down", "unhappy", "miserable"],
        "anxious": ["anxious", "worried", "nervous", "stressed", "scared", "afraid"],
        "happy": ["happy", "excited", "great", "awesome", "amazing", "wonderful", "fantastic"],
        "disgusted": ["disgusted", "disgusting", "gross", "repulsive", "revolting"],
        "surprised": ["amazed", "surprised", "unexpected", "stunned", "astonished", "shocked"],
        "angry": ["furious", "rage", "livid", "outraged", "infuriated"],
        "fearful": ["terrified", "frightened", "petrified", "horrified"],
    }

    # Coarse affective coordinates used for disagreement fallback.
    # Range convention: valence [-1, 1], arousal [0, 1].
    EMOTION_VA = {
        "happy": (0.8, 0.7),
        "surprised": (0.3, 0.9),
        "angry": (-0.7, 0.85),
        "frustrated": (-0.5, 0.7),
        "fearful": (-0.8, 0.9),
        "anxious": (-0.6, 0.8),
        "sad": (-0.8, 0.3),
        "disgusted": (-0.7, 0.6),
        "neutral": (0.0, 0.4),
    }

    BASE_PROMPT = (
        "You are a helpful, respectful, conversational assistant. "
        "Do not use emojis, emoticons, or decorative symbols in your responses. "
        "Write clean, readable responses with normal punctuation and paragraph breaks. "
        "If there are multiple points, format them as short bullet points or numbered steps. "
        "Keep each response concise by default (about 3-6 sentences) unless the user asks for detail."
    )

    EMOTION_PROMPTS = {
        "frustrated": " The user seems frustrated. Respond calmly, patiently, and with clear explanations.",
        "sad": " The user seems sad. Respond warmly, empathetically, and supportively while staying practical.",
        "anxious": " The user seems anxious. Keep the response calm, grounding, and reassuring.",
        "happy": " The user seems happy. Match the positive energy while staying helpful and engaged.",
        "disgusted": " The user seems disgusted. Acknowledge their feelings and gently redirect if appropriate.",
        "surprised": " The user seems surprised. Acknowledge their surprise and respond with a steady demeanor.",
        "angry": " The user seems angry. Stay calm, do not escalate, and address their concerns directly.",
        "fearful": " The user seems fearful. Be reassuring, gentle, and provide clear information.",
        "neutral": "",
    }

    AGE_PROMPTS = {
        "child": " Use simple language appropriate for children. Be friendly and patient.",
        "teen": " Be relatable but respectful. Avoid being condescending.",
        "adult": "",
        "senior": " Be clear and patient. Offer to explain things if needed.",
    }

    def __init__(self):
        self.external_emotions: Dict[str, ExternalEmotion] = {}

    def detect_text_emotion(self, text: str) -> str:
        text_lower = text.lower()

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion

        return "neutral"

    @staticmethod
    def _normalize_emotion_label(label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        label = label.strip().lower()
        aliases = {
            "anger": "angry",
            "disgust": "disgusted",
            "fear": "fearful",
            "surprise": "surprised",
            "calm": "neutral",
        }
        return aliases.get(label, label)

    @staticmethod
    def _clip_confidence(conf: Optional[float]) -> float:
        if conf is None:
            return 0.0
        return max(0.0, min(1.0, float(conf)))

    @staticmethod
    def _clip_valence(valence: Optional[float]) -> Optional[float]:
        if valence is None:
            return None
        return max(-1.0, min(1.0, float(valence)))

    @staticmethod
    def _clip_arousal(arousal: Optional[float]) -> Optional[float]:
        if arousal is None:
            return None
        return max(0.0, min(1.0, float(arousal)))

    def _nearest_emotion_from_valence_arousal(self, valence: float, arousal: float) -> str:
        best_emotion = "neutral"
        best_distance = float("inf")
        for emotion, (ev, ea) in self.EMOTION_VA.items():
            dist = (ev - valence) ** 2 + (ea - arousal) ** 2
            if dist < best_distance:
                best_distance = dist
                best_emotion = emotion
        return best_emotion

    def _fuse_modalities(
        self,
        face_emotion: Optional[str],
        voice_emotion: Optional[str],
        face_confidence: float,
        voice_confidence: float,
        face_valence: Optional[float],
        face_arousal: Optional[float],
        voice_valence: Optional[float],
        voice_arousal: Optional[float],
    ) -> Tuple[str, float]:
        face_emotion = self._normalize_emotion_label(face_emotion)
        voice_emotion = self._normalize_emotion_label(voice_emotion)

        face_confidence = self._clip_confidence(face_confidence)
        voice_confidence = self._clip_confidence(voice_confidence)
        face_valence = self._clip_valence(face_valence)
        face_arousal = self._clip_arousal(face_arousal)
        voice_valence = self._clip_valence(voice_valence)
        voice_arousal = self._clip_arousal(voice_arousal)

        # Rule requested by UI: final prediction comes from the modality with greater confidence.
        if face_emotion and voice_emotion:
            if face_confidence >= voice_confidence:
                return face_emotion, face_confidence
            return voice_emotion, voice_confidence

        if face_emotion:
            return face_emotion, face_confidence

        if voice_emotion:
            return voice_emotion, voice_confidence

        return "neutral", max(face_confidence, voice_confidence, 0.0)

    def update_external_emotion(
        self,
        user_id: str,
        face_emotion: Optional[str],
        voice_emotion: Optional[str],
        confidence: float = 0.0,
        face_confidence: Optional[float] = None,
        voice_confidence: Optional[float] = None,
        face_valence: Optional[float] = None,
        face_arousal: Optional[float] = None,
        voice_valence: Optional[float] = None,
        voice_arousal: Optional[float] = None,
    ):
        fallback_conf = self._clip_confidence(confidence)
        face_conf = self._clip_confidence(face_confidence if face_confidence is not None else fallback_conf)
        voice_conf = self._clip_confidence(voice_confidence if voice_confidence is not None else fallback_conf)

        fused_emotion, fused_conf = self._fuse_modalities(
            face_emotion=face_emotion,
            voice_emotion=voice_emotion,
            face_confidence=face_conf,
            voice_confidence=voice_conf,
            face_valence=face_valence,
            face_arousal=face_arousal,
            voice_valence=voice_valence,
            voice_arousal=voice_arousal,
        )

        self.external_emotions[user_id] = ExternalEmotion(
            face_emotion=self._normalize_emotion_label(face_emotion),
            voice_emotion=self._normalize_emotion_label(voice_emotion),
            face_confidence=face_conf,
            voice_confidence=voice_conf,
            face_valence=self._clip_valence(face_valence),
            face_arousal=self._clip_arousal(face_arousal),
            voice_valence=self._clip_valence(voice_valence),
            voice_arousal=self._clip_arousal(voice_arousal),
            confidence=max(fused_conf, fallback_conf),
            fused_emotion=fused_emotion,
            timestamp=datetime.now(),
        )

    def get_external_emotion(self, user_id: str) -> Optional[ExternalEmotion]:
        external = self.external_emotions.get(user_id)
        if external:
            age_seconds = (datetime.now() - external.timestamp).total_seconds()
            if age_seconds < 30:
                return external
        return None

    def combine_emotions(
        self,
        text_emotion: str,
        provided_emotion: Optional[str] = None,
        confidence: float = 0.0,
        user_id: Optional[str] = None,
    ) -> str:
        external = None
        if user_id:
            external = self.get_external_emotion(user_id)

        # Prefer recent multimodal emotion whenever it is at least moderately
        # confident so response tone follows face/voice state in real-time.
        if external and external.confidence >= 0.35:
            if external.fused_emotion and external.fused_emotion != "neutral":
                return external.fused_emotion
            if external.face_emotion and external.face_emotion != "neutral":
                return external.face_emotion
            if external.voice_emotion and external.voice_emotion != "neutral":
                return external.voice_emotion

        if provided_emotion and provided_emotion != "neutral" and confidence > 0.5:
            return provided_emotion

        return text_emotion

    def get_age_group(self, age: int) -> str:
        if age < 13:
            return "child"
        elif age < 20:
            return "teen"
        elif age < 65:
            return "adult"
        else:
            return "senior"

    def get_adapted_prompt(self, emotion: str, age: int) -> str:
        base = self.BASE_PROMPT
        emotion_modifier = self.EMOTION_PROMPTS.get(emotion, "")
        age_group = self.get_age_group(age)
        age_modifier = self.AGE_PROMPTS.get(age_group, "")

        return base + emotion_modifier + age_modifier

    def get_tone_description(self, emotion: str) -> str:
        descriptions = {
            "frustrated": "calm and patient",
            "sad": "warm and supportive",
            "anxious": "grounding and reassuring",
            "happy": "positive and engaged",
            "disgusted": "neutral and redirecting",
            "surprised": "steady and acknowledging",
            "angry": "calm and direct",
            "fearful": "gentle and reassuring",
            "neutral": "helpful and conversational",
        }
        return descriptions.get(emotion, "helpful")

    def get_tts_speaker(self, emotion: str) -> str:
        # Map emotion to available VibeVoice preset speakers.
        # Keep "Carter" for neutral/calm and "Sophia" for warm supportive tone.
        speaker_map = {
            "frustrated": "Carter",
            "sad": "Sophia",
            "anxious": "Sophia",
            "happy": "Emma",
            "disgusted": "Carter",
            "surprised": "Emma",
            "angry": "Carter",
            "fearful": "Sophia",
            "neutral": "Carter",
        }
        return speaker_map.get(emotion, "Carter")

    def add_tts_tone_hint(self, text: str, emotion: str) -> str:
        # VibeVoice does not require SSML. A short textual style hint can steer prosody.
        hint_map = {
            "frustrated": "Say this calmly and clearly: ",
            "sad": "Say this warmly and gently: ",
            "anxious": "Say this in a calm, reassuring tone: ",
            "happy": "Say this in an upbeat, friendly tone: ",
            "disgusted": "Say this in a neutral, steady tone: ",
            "surprised": "Say this with mild surprise but clear pacing: ",
            "angry": "Say this in a calm and steady tone: ",
            "fearful": "Say this gently and reassuringly: ",
            "neutral": "",
        }
        prefix = hint_map.get(emotion, "")
        return f"{prefix}{text}" if prefix else text
