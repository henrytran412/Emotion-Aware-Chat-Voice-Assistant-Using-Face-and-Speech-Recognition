/**
 * Browser-based Face Emotion Detection
 * Uses TensorFlow.js for lightweight emotion inference from webcam
 */

class EmotionDetector {
    constructor() {
        this.model = null;
        this.isRunning = false;
        this.currentEmotion = 'neutral';
        this.confidence = 0.45;
        this.videoElement = null;
        this.onEmotionChange = null;
        this.lastEmitAt = 0;
        this.lastErrorAt = 0;
        this.lastSmoothed = {
            happy: 0,
            sad: 0,
            surprised: 0,
            angry: 0,
        };

        // Simple emotion classifier based on face landmarks
        // Maps facial features to emotions
        this.emotionThresholds = {
            happy: { mouthOpen: 0.3, eyebrowRaise: 0.1 },
            sad: { mouthCornerDown: 0.2, eyebrowFurrow: 0.15 },
            surprised: { mouthOpen: 0.5, eyebrowRaise: 0.3 },
            angry: { eyebrowFurrow: 0.25, mouthTense: 0.2 },
        };
    }

    async initialize() {
        try {
            // Load face landmarks model
            this.model = await faceLandmarksDetection.createDetector(
                faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
                {
                    runtime: 'tfjs',
                    refineLandmarks: true,
                    maxFaces: 1,
                }
            );
            console.log('Face emotion detector initialized');
            return true;
        } catch (error) {
            console.error('Failed to initialize emotion detector:', error);
            return false;
        }
    }

    setVideoElement(video) {
        this.videoElement = video;
    }

    start() {
        if (!this.model || !this.videoElement) {
            console.warn('Emotion detector not ready');
            return;
        }
        this.isRunning = true;
        this.detectLoop();
    }

    stop() {
        this.isRunning = false;
    }

    async detectLoop() {
        if (!this.isRunning) return;

        try {
            const faces = await this.model.estimateFaces(this.videoElement);

            if (faces.length > 0) {
                const emotion = this.analyzeEmotion(faces[0]);
                const now = Date.now();
                const changed = emotion.emotion !== this.currentEmotion;
                const periodicUpdate = now - this.lastEmitAt > 1400;
                if (changed || periodicUpdate) {
                    this.currentEmotion = emotion.emotion;
                    this.confidence = emotion.confidence;
                    this.lastEmitAt = now;
                    if (this.onEmotionChange) {
                        this.onEmotionChange(emotion);
                    }
                }
            }
        } catch (error) {
            const now = Date.now();
            if (now - this.lastErrorAt > 3000) {
                this.lastErrorAt = now;
                console.warn('Emotion detector loop error:', error?.message || error);
            }
        }

        // Run detection every 500ms to save CPU
        if (this.isRunning) {
            setTimeout(() => this.detectLoop(), 500);
        }
    }

    analyzeEmotion(face) {
        const keypoints = face.keypoints;

        // Get key facial landmarks
        const landmarks = this.extractLandmarks(keypoints);

        // Calculate facial features
        const features = this.calculateFeatures(landmarks);

        // Classify emotion based on features
        return this.classifyEmotion(features);
    }

    extractLandmarks(keypoints) {
        // Key landmark indices for MediaPipe Face Mesh
        const indices = {
            leftEyeOuter: 33,
            leftEyeInner: 133,
            rightEyeOuter: 362,
            rightEyeInner: 263,
            leftEyebrowOuter: 70,
            leftEyebrowInner: 107,
            rightEyebrowOuter: 300,
            rightEyebrowInner: 336,
            noseTip: 1,
            mouthLeft: 61,
            mouthRight: 291,
            mouthTop: 13,
            mouthBottom: 14,
            upperLipMid: 0,
            lowerLipMid: 17,
            chin: 152,
        };

        const landmarks = {};
        for (const [name, index] of Object.entries(indices)) {
            if (keypoints[index]) {
                landmarks[name] = keypoints[index];
            }
        }
        return landmarks;
    }

    calculateFeatures(landmarks) {
        const features = {
            mouthOpenness: 0,
            mouthWidth: 0,
            mouthSmile: 0,
            mouthFrown: 0,
            eyebrowHeight: 0,
            browFurrow: 0,
            eyeOpenness: 0.5,
        };

        try {
            // Mouth openness (vertical)
            if (landmarks.mouthTop && landmarks.mouthBottom) {
                const mouthHeight = Math.abs(landmarks.mouthTop.y - landmarks.mouthBottom.y);
                features.mouthOpenness = mouthHeight / 50; // Normalize
            }

            // Mouth width
            if (landmarks.mouthLeft && landmarks.mouthRight) {
                const mouthWidth = Math.abs(landmarks.mouthLeft.x - landmarks.mouthRight.x);
                features.mouthWidth = mouthWidth / 100; // Normalize

                if (landmarks.upperLipMid && landmarks.lowerLipMid) {
                    const lipMidY = (landmarks.upperLipMid.y + landmarks.lowerLipMid.y) / 2;
                    const cornerY = (landmarks.mouthLeft.y + landmarks.mouthRight.y) / 2;
                    const smileDelta = (lipMidY - cornerY) / 25;
                    features.mouthSmile = Math.max(0, smileDelta);
                    features.mouthFrown = Math.max(0, -smileDelta);
                }
            }

            // Eyebrow height (relative to eyes)
            if (landmarks.leftEyebrowInner && landmarks.leftEyeInner) {
                const eyebrowDist = landmarks.leftEyeInner.y - landmarks.leftEyebrowInner.y;
                features.eyebrowHeight = eyebrowDist / 30; // Normalize
            }

            if (landmarks.leftEyebrowInner && landmarks.rightEyebrowInner) {
                const browInnerDist = Math.abs(landmarks.leftEyebrowInner.x - landmarks.rightEyebrowInner.x);
                features.browFurrow = Math.max(0, (0.24 - browInnerDist / 100));
            }

            // Eye openness
            if (landmarks.leftEyeOuter && landmarks.leftEyeInner) {
                features.eyeOpenness = 0.5; // Simplified
            }
        } catch (e) {
            // Use defaults if landmarks missing
        }

        return features;
    }

    classifyEmotion(features) {
        const rawScores = {
            happy: (features.mouthSmile * 1.25) + (features.mouthWidth * 0.35),
            surprised: (features.mouthOpenness * 1.15) + (features.eyebrowHeight * 0.7),
            sad: (features.mouthFrown * 1.1) + (Math.max(0, 0.35 - features.eyebrowHeight) * 0.9),
            angry: (features.browFurrow * 1.2) + (features.mouthFrown * 0.6),
        };

        const alpha = 0.45;
        for (const key of Object.keys(rawScores)) {
            this.lastSmoothed[key] = (1 - alpha) * this.lastSmoothed[key] + alpha * rawScores[key];
        }

        const entries = Object.entries(this.lastSmoothed).sort((a, b) => b[1] - a[1]);
        const [bestEmotion, bestScore] = entries[0];
        const secondScore = entries[1][1];
        const margin = bestScore - secondScore;

        if (bestScore < 0.22 || margin < 0.05) {
            return { emotion: 'neutral', confidence: 0.45 };
        }

        const confidence = Math.max(0.5, Math.min(0.95, 0.55 + bestScore * 0.35 + margin * 0.4));
        return { emotion: bestEmotion, confidence };
    }

    getCurrentEmotion() {
        return {
            emotion: this.currentEmotion,
            confidence: this.confidence,
        };
    }
}

// Global instance
window.emotionDetector = new EmotionDetector();
