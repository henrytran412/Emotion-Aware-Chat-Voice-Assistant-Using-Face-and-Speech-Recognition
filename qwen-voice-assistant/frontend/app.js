/**
 * Emotion-Aware AI Assistant - Frontend Application
 */

let API_BASE = localStorage.getItem('api_base') || '';

let currentUser = null;
let mediaStream = null;
let cameraActive = false;
let micActive = false;
let faceEmotionEnabled = localStorage.getItem('face_emotion_enabled') !== 'false';
let detectedEmotion = { emotion: 'neutral', confidence: 0 };
let chatMode = localStorage.getItem('echo_chat_mode') || '';
let voiceSessionActive = false;
let voiceSessionBusy = false;
let voiceRecognition = null;
let voiceFinalBuffer = '';
let voiceInterimBuffer = '';
let voiceSilenceTimer = null;
let voiceStopRequested = false;
let cameraDragEnabled = false;
let cameraDragState = null;
let emotionSyncTimer = null;
let bridgeEmotionActive = false;
let bridgeEmotionStale = false;
let bridgeFrameTimer = null;
let cameraFeedMode = null;

const authScreen = document.getElementById('auth-screen');
const chatScreen = document.getElementById('chat-screen');
const registerForm = document.getElementById('register-form');
const userNameInput = document.getElementById('user-name');
const userBirthdayInput = document.getElementById('user-birthday');
const displayName = document.getElementById('display-name');
const userAvatar = document.getElementById('user-avatar');
const emotionBadge = document.getElementById('emotion-badge');
const modeBadge = document.getElementById('mode-badge');
const chatMain = document.querySelector('.chat-main');
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const voiceInputBtn = document.getElementById('voice-input-btn');
const cameraToggle = document.getElementById('camera-toggle');
const micToggle = document.getElementById('mic-toggle');
const cameraPreview = document.getElementById('camera-preview');
const cameraVideo = document.getElementById('camera-video');
const cameraBridgeImage = document.getElementById('camera-bridge-image');
const cameraStatus = cameraPreview.querySelector('.camera-status');
const typingIndicator = document.getElementById('typing-indicator');
const ttsAudio = document.getElementById('tts-audio');
const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const serverUrlInput = document.getElementById('server-url');
const enableFaceEmotionCheckbox = document.getElementById('enable-face-emotion');
const modeSwitchBtn = document.getElementById('mode-switch-btn');
const modeSwitchLabel = document.getElementById('mode-switch-label');
const modeTextBtn = document.getElementById('mode-text-btn');
const modeVoiceBtn = document.getElementById('mode-voice-btn');
const chatInputArea = document.getElementById('chat-input-area');
const voiceModeStage = document.getElementById('voice-mode-stage');
const voiceOrb = document.getElementById('voice-orb');
const voiceStatus = document.getElementById('voice-status');
const voiceSessionBtn = document.getElementById('voice-session-btn');
const startChatBtn = document.getElementById('start-chat-btn');
const voiceEmotionPill = document.getElementById('voice-emotion-pill');
const emotionSourceLabel = document.getElementById('emotion-source-label');
const facePrediction = document.getElementById('face-prediction');
const voicePrediction = document.getElementById('voice-prediction');
const finalPrediction = document.getElementById('final-prediction');

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    registerForm.addEventListener('submit', handleRegister);
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', handleInputKeydown);
    messageInput.addEventListener('input', autoResizeTextarea);
    cameraToggle.addEventListener('click', toggleCamera);
    micToggle.addEventListener('click', toggleMicrophone);
    voiceInputBtn.addEventListener('click', startVoiceInput);
    settingsBtn.addEventListener('click', openSettings);
    modeSwitchBtn.addEventListener('click', toggleChatModeFromHeader);
    modeTextBtn.addEventListener('click', () => setChatMode('text'));
    modeVoiceBtn.addEventListener('click', () => setChatMode('voice'));
    voiceSessionBtn.addEventListener('click', toggleVoiceSession);

    const today = new Date().toISOString().split('T')[0];
    userBirthdayInput.max = today;

    if (window.emotionDetector && faceEmotionEnabled) {
        const initialized = await window.emotionDetector.initialize();
        if (initialized) {
            window.emotionDetector.onEmotionChange = (emotion) => {
                detectedEmotion = emotion;
                updateEmotionBadge(emotion.emotion);
                updateVoiceOrb(emotion.emotion, emotion.confidence);
                updateVoiceEmotionPill(emotion.emotion, emotion.confidence);
                sendExternalEmotionUpdate(emotion);
            };
        }
    }

    serverUrlInput.value = API_BASE;
    enableFaceEmotionCheckbox.checked = faceEmotionEnabled;
    renderModeButtons();
}

function openSettings() {
    settingsModal.classList.remove('hidden');
    serverUrlInput.value = API_BASE;
    enableFaceEmotionCheckbox.checked = faceEmotionEnabled;
}

function closeSettings() {
    settingsModal.classList.add('hidden');
}

function saveSettings() {
    const newApiBase = serverUrlInput.value.trim();
    const newFaceEmotion = enableFaceEmotionCheckbox.checked;

    localStorage.setItem('api_base', newApiBase);
    localStorage.setItem('face_emotion_enabled', newFaceEmotion);

    API_BASE = newApiBase;
    faceEmotionEnabled = newFaceEmotion;

    if (faceEmotionEnabled && window.emotionDetector && !window.emotionDetector.model) {
        window.emotionDetector.initialize();
    }

    closeSettings();
}

window.closeSettings = closeSettings;
window.saveSettings = saveSettings;

async function handleRegister(e) {
    e.preventDefault();

    const name = userNameInput.value.trim();
    const birthday = userBirthdayInput.value;

    if (!name || !birthday) {
        alert('Please fill in all fields');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/users`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, birthday }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Registration failed');
        }

        currentUser = await response.json();
        localStorage.setItem('emotionAI_user', JSON.stringify(currentUser));
        showChatScreen();
    } catch (error) {
        if (error.message.includes('fetch')) {
            currentUser = {
                user_id: `local-${Date.now()}`,
                name,
                age: calculateAge(birthday),
                created_at: new Date().toISOString(),
            };
            localStorage.setItem('emotionAI_user', JSON.stringify(currentUser));
            showChatScreen();
        } else {
            alert(error.message);
        }
    }
}

function calculateAge(birthday) {
    const birth = new Date(birthday);
    const today = new Date();
    let age = today.getFullYear() - birth.getFullYear();
    const monthDiff = today.getMonth() - birth.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
        age -= 1;
    }
    return age;
}

function showChatScreen() {
    if (!chatMode) {
        alert('Please choose a chat type first.');
        return;
    }

    authScreen.classList.remove('active');
    chatScreen.classList.add('active');

    displayName.textContent = currentUser.name;
    userAvatar.textContent = currentUser.name.charAt(0).toUpperCase();
    applyChatModeUI();
    startEmotionSyncLoop();

    if (chatMode === 'text') {
        messageInput.focus();
    }
}

function setChatMode(mode) {
    chatMode = mode === 'voice' ? 'voice' : 'text';
    localStorage.setItem('echo_chat_mode', chatMode);
    renderModeButtons();
    applyChatModeUI();
}

function toggleChatModeFromHeader() {
    const nextMode = chatMode === 'voice' ? 'text' : 'voice';
    setChatMode(nextMode);
}

function renderModeButtons() {
    modeTextBtn.classList.toggle('active', chatMode === 'text');
    modeVoiceBtn.classList.toggle('active', chatMode === 'voice');
    startChatBtn.disabled = !chatMode;

    if (chatMode === 'voice') {
        modeSwitchLabel.textContent = 'Text';
    } else {
        modeSwitchLabel.textContent = 'Voice';
    }
}

function applyChatModeUI() {
    const onChatScreen = chatScreen.classList.contains('active');
    if (!onChatScreen) {
        return;
    }

    if (chatMode === 'voice') {
        chatMain.classList.add('voice-active');
        chatInputArea.classList.add('hidden');
        voiceModeStage.classList.remove('hidden');
        modeBadge.textContent = 'Voice mode (VibeVoice)';
        setVoiceStageState('idle');
        updateVoiceStatus('Ready. Press Start Voice Chat.');
        updateVoiceEmotionPill(detectedEmotion.emotion, detectedEmotion.confidence);
    } else {
        stopVoiceSession();
        chatMain.classList.remove('voice-active');
        chatInputArea.classList.remove('hidden');
        voiceModeStage.classList.add('hidden');
        modeBadge.textContent = 'Text mode';
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || !currentUser) return;

    appendMessage('user', message);
    messageInput.value = '';
    autoResizeTextarea();

    await requestAssistantResponse(message, false);
}

async function requestAssistantResponse(message, fromVoice) {
    typingIndicator.classList.remove('hidden');
    scrollToBottom();

    // Refresh bridge emotion just before sending so LLM tone uses the latest
    // face/voice fusion from backend when available.
    let bridgeFresh = false;
    if (currentUser?.user_id) {
        bridgeFresh = await fetchExternalEmotion();
    }

    const useBrowserEmotion = (
        cameraFeedMode === 'browser' &&
        cameraActive &&
        faceEmotionEnabled &&
        window.emotionDetector &&
        window.emotionDetector.isRunning &&
        !bridgeFresh
    );

    const currentEmotion = useBrowserEmotion
        ? window.emotionDetector.getCurrentEmotion()
        : detectedEmotion;

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                user_id: currentUser.user_id,
                emotion: currentEmotion.emotion,
                emotion_confidence: currentEmotion.confidence,
            }),
        });

        if (!response.ok) {
            throw new Error('Chat request failed');
        }

        const data = await response.json();
        typingIndicator.classList.add('hidden');

        updateEmotionBadge(data.detected_emotion);
        updateVoiceOrb(data.detected_emotion, currentEmotion.confidence || 0.5);
        updateVoiceEmotionPill(data.detected_emotion, currentEmotion.confidence || 0.5);

        appendMessage('assistant', data.response, {
            tone: data.adapted_tone,
            audioUrl: data.audio_url,
        });

        if (data.audio_url) {
            const url = `${API_BASE}${data.audio_url}`;
            if (fromVoice) {
                updateVoiceStatus('ECHO is speaking with VibeVoice...');
                await playAudioAndWait(url);
            } else {
                playAudio(url);
            }
        }
    } catch (error) {
        typingIndicator.classList.add('hidden');

        let errorMsg = "I'm having trouble connecting to the server.";
        if (!API_BASE) {
            errorMsg += ' Make sure the backend is running on this machine.';
        } else {
            errorMsg += ` Check if the server at ${API_BASE} is running.`;
        }
        errorMsg += ' Click the settings icon to configure the server URL.';

        appendMessage('assistant', errorMsg, { tone: 'helpful' });
        if (fromVoice) {
            updateVoiceStatus('Connection issue. Check server settings.');
        }
    }
}

function appendMessage(role, content, options = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    let avatarHtml = '';
    if (role === 'assistant') {
        avatarHtml = '<div class="assistant-avatar">EC</div>';
    }

    let metaHtml = '';
    if (options.tone) {
        metaHtml += `<span class="tone-indicator">${options.tone}</span>`;
    }
    if (options.audioUrl) {
        metaHtml += `
            <button class="audio-btn" onclick="playAudio('${options.audioUrl}')" title="Play audio">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
            </button>
        `;
    }

    messageDiv.innerHTML = `
        ${avatarHtml}
        <div>
            <div class="message-bubble">${escapeHtml(content)}</div>
            ${metaHtml ? `<div class="message-meta">${metaHtml}</div>` : ''}
        </div>
    `;

    const welcome = chatMessages.querySelector('.welcome-message');
    if (welcome && role === 'user') {
        welcome.remove();
    }

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function updateEmotionBadge(emotion) {
    emotionBadge.textContent = emotion;
    emotionBadge.dataset.emotion = emotion;
}

function updateVoiceOrb(emotion, confidence = 0.5) {
    voiceOrb.dataset.emotion = emotion || 'neutral';
    const scale = 0.8 + Math.min(Math.max(confidence, 0), 1) * 0.8;
    voiceOrb.style.setProperty('--wave-scale', scale.toFixed(2));
}

function updateVoiceEmotionPill(emotion, confidence = 0) {
    if (!voiceEmotionPill) return;
    const em = emotion || 'neutral';
    voiceEmotionPill.dataset.emotion = em;
    voiceEmotionPill.textContent = `${capitalize(em)} ${(Math.max(0, Math.min(1, confidence)) * 100).toFixed(0)}%`;
}

function updateEmotionSourceLabel(sourceText, state = 'browser') {
    if (!emotionSourceLabel) return;
    emotionSourceLabel.textContent = `Source: ${sourceText}`;
    emotionSourceLabel.classList.remove('bridge-active', 'bridge-stale');
    if (state === 'bridge-active') {
        emotionSourceLabel.classList.add('bridge-active');
    }
    if (state === 'bridge-stale') {
        emotionSourceLabel.classList.add('bridge-stale');
    }
}

function formatPrediction(emotion, confidence) {
    const em = emotion || 'none';
    const conf = Math.max(0, Math.min(1, Number(confidence || 0)));
    return `${capitalize(em)} ${(conf * 100).toFixed(0)}%`;
}

function updatePredictionPanel(faceEmotion, faceConfidence, voiceEmotion, voiceConfidence, finalEmotion, finalConfidence) {
    if (facePrediction) {
        facePrediction.textContent = formatPrediction(faceEmotion || 'none', faceConfidence || 0);
    }
    if (voicePrediction) {
        voicePrediction.textContent = formatPrediction(voiceEmotion || 'none', voiceConfidence || 0);
    }
    if (finalPrediction) {
        finalPrediction.textContent = formatPrediction(finalEmotion || 'neutral', finalConfidence || 0);
    }
}

function capitalize(text) {
    if (!text) return '';
    return text.charAt(0).toUpperCase() + text.slice(1);
}

async function sendExternalEmotionUpdate(emotion) {
    if (!currentUser?.user_id || !emotion?.emotion) return;
    try {
        await fetch(`${API_BASE}/api/emotion`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: currentUser.user_id,
                face_emotion: emotion.emotion,
                face_confidence: emotion.confidence,
                confidence: emotion.confidence,
            }),
        });
    } catch (error) {
        // Keep UI responsive if backend is temporarily unavailable.
    }
}

async function fetchExternalEmotion() {
    if (!currentUser?.user_id) return;
    try {
        const response = await fetch(`${API_BASE}/api/emotion/${encodeURIComponent(currentUser.user_id)}`);
        if (!response.ok) {
            bridgeEmotionActive = false;
            bridgeEmotionStale = false;
            updateEmotionSourceLabel('Browser detector', 'browser');
            return false;
        }

        const data = await response.json();
        if (!data?.fused_emotion) {
            bridgeEmotionActive = false;
            bridgeEmotionStale = false;
            updateEmotionSourceLabel('Browser detector', 'browser');
            return false;
        }

        const ageSeconds = Number(data.age_seconds || 0);
        const stale = ageSeconds > 3.0;
        bridgeEmotionActive = !stale;
        bridgeEmotionStale = stale;

        const faceEmotion = data.face_emotion || null;
        const voiceEmotion = data.voice_emotion || null;
        const faceConfidence = Number(data.face_confidence || 0);
        const voiceConfidence = Number(data.voice_confidence || 0);

        let finalEmotion = data.fused_emotion || 'neutral';
        let finalConfidence = Number(data.confidence || 0);

        if (faceEmotion || voiceEmotion) {
            if (faceConfidence >= voiceConfidence) {
                finalEmotion = faceEmotion || voiceEmotion || finalEmotion;
                finalConfidence = Math.max(faceConfidence, voiceConfidence, finalConfidence);
            } else {
                finalEmotion = voiceEmotion || faceEmotion || finalEmotion;
                finalConfidence = Math.max(voiceConfidence, faceConfidence, finalConfidence);
            }
        }

        updatePredictionPanel(
            faceEmotion,
            faceConfidence,
            voiceEmotion,
            voiceConfidence,
            finalEmotion,
            finalConfidence,
        );

        const extEmotion = {
            emotion: finalEmotion,
            confidence: finalConfidence,
        };
        detectedEmotion = extEmotion;
        updateEmotionBadge(extEmotion.emotion);
        updateVoiceOrb(extEmotion.emotion, extEmotion.confidence);
        updateVoiceEmotionPill(extEmotion.emotion, extEmotion.confidence);
        if (stale) {
            updateEmotionSourceLabel('PC bridge (stale)', 'bridge-stale');
        } else {
            updateEmotionSourceLabel('PC bridge', 'bridge-active');
        }
        return !stale;
    } catch (error) {
        bridgeEmotionActive = false;
        bridgeEmotionStale = false;
        updatePredictionPanel(null, 0, null, 0, detectedEmotion.emotion, detectedEmotion.confidence);
        updateEmotionSourceLabel('Browser detector', 'browser');
        return false;
    }
}

function setVoiceStageState(state) {
    voiceOrb.dataset.state = state;
}

function updateVoiceStatus(statusText) {
    voiceStatus.textContent = statusText;
}

function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = `${Math.min(messageInput.scrollHeight, 120)}px`;
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function getBridgeFrameUrl() {
    if (!currentUser?.user_id) return null;
    return `${API_BASE}/api/bridge-frame/${encodeURIComponent(currentUser.user_id)}?t=${Date.now()}`;
}

function setCameraStatus(text) {
    if (cameraStatus) {
        cameraStatus.textContent = text;
    }
}

async function startBridgeCameraFeed() {
    if (!cameraBridgeImage || !currentUser?.user_id) return false;

    stopBridgeCameraFeed();

    const initialUrl = getBridgeFrameUrl();
    if (!initialUrl) return false;

    const loaded = await new Promise((resolve) => {
        let settled = false;

        const finish = (ok) => {
            if (settled) return;
            settled = true;
            resolve(ok);
        };

        cameraBridgeImage.onload = () => finish(true);
        cameraBridgeImage.onerror = () => finish(false);
        cameraBridgeImage.src = initialUrl;

        setTimeout(() => finish(false), 1200);
    });

    if (!loaded) {
        cameraBridgeImage.onload = null;
        cameraBridgeImage.onerror = null;
        cameraBridgeImage.removeAttribute('src');
        return false;
    }

    cameraPreview.classList.add('bridge-mode');
    cameraBridgeImage.classList.remove('hidden');
    cameraFeedMode = 'bridge';
    setCameraStatus('Bridge Camera Active');

    cameraBridgeImage.onload = null;
    cameraBridgeImage.onerror = null;

    bridgeFrameTimer = setInterval(() => {
        const nextUrl = getBridgeFrameUrl();
        if (!nextUrl) return;
        cameraBridgeImage.src = nextUrl;
    }, 300);

    return true;
}

function stopBridgeCameraFeed() {
    if (bridgeFrameTimer) {
        clearInterval(bridgeFrameTimer);
        bridgeFrameTimer = null;
    }
    if (cameraBridgeImage) {
        cameraBridgeImage.onload = null;
        cameraBridgeImage.onerror = null;
        cameraBridgeImage.removeAttribute('src');
        cameraBridgeImage.classList.add('hidden');
    }
    cameraPreview.classList.remove('bridge-mode');
    if (cameraFeedMode === 'bridge') {
        cameraFeedMode = null;
    }
}

async function toggleCamera() {
    if (cameraActive) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    cameraPreview.classList.remove('hidden');
    cameraToggle.classList.add('active');
    enableCameraDrag();

    const hasBridgeFeed = await startBridgeCameraFeed();
    if (hasBridgeFeed) {
        cameraActive = true;
        if (window.emotionDetector) {
            window.emotionDetector.stop();
        }
        startEmotionSyncLoop();
        return;
    }

    cameraPreview.classList.remove('bridge-mode');
    setCameraStatus('Browser Camera Active');

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 320, height: 240 },
        });

        cameraVideo.onloadeddata = null;
        cameraVideo.onloadedmetadata = null;
        cameraVideo.srcObject = mediaStream;

        try {
            await cameraVideo.play();
        } catch (error) {
            // Some browsers may require user gesture; continue and rely on loaded events.
        }

        cameraFeedMode = 'browser';
        cameraActive = true;

        if (faceEmotionEnabled && window.emotionDetector) {
            if (!window.emotionDetector.model) {
                await window.emotionDetector.initialize();
            }
            window.emotionDetector.setVideoElement(cameraVideo);

            const startDetector = () => {
                window.emotionDetector.start();
                const current = window.emotionDetector.getCurrentEmotion();
                detectedEmotion = current;
                updateEmotionBadge(current.emotion);
                updateVoiceOrb(current.emotion, current.confidence);
                updateVoiceEmotionPill(current.emotion, current.confidence);
                startEmotionSyncLoop();
            };

            if (cameraVideo.readyState >= 2) {
                startDetector();
            } else {
                cameraVideo.onloadedmetadata = startDetector;
                cameraVideo.onloadeddata = startDetector;
            }
        } else {
            startEmotionSyncLoop();
        }
    } catch (error) {
        stopBridgeCameraFeed();
        cameraPreview.classList.add('hidden');
        cameraToggle.classList.remove('active');
        cameraActive = false;
        cameraFeedMode = null;
        alert('Could not access camera. Please check permissions.');
    }
}

function stopCamera() {
    stopEmotionSyncLoop();
    stopBridgeCameraFeed();

    if (window.emotionDetector) {
        window.emotionDetector.stop();
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach((track) => track.stop());
        mediaStream = null;
    }
    cameraVideo.srcObject = null;
    cameraPreview.classList.add('hidden');
    cameraToggle.classList.remove('active');
    cameraActive = false;
    cameraFeedMode = null;
    setCameraStatus('Camera Active');
}

function startEmotionSyncLoop() {
    stopEmotionSyncLoop();
    emotionSyncTimer = setInterval(async () => {
        const bridgeFresh = await fetchExternalEmotion();
        if (!bridgeFresh && faceEmotionEnabled && cameraActive && window.emotionDetector && window.emotionDetector.isRunning) {
            const current = window.emotionDetector.getCurrentEmotion();
            detectedEmotion = current;
            updateEmotionBadge(current.emotion);
            updateVoiceOrb(current.emotion, current.confidence);
            updateVoiceEmotionPill(current.emotion, current.confidence);
            sendExternalEmotionUpdate(current);
            updateEmotionSourceLabel('Browser detector', 'browser');
        }
    }, 1400);
}

function stopEmotionSyncLoop() {
    if (emotionSyncTimer) {
        clearInterval(emotionSyncTimer);
        emotionSyncTimer = null;
    }
}

async function toggleMicrophone() {
    micActive = !micActive;
    micToggle.classList.toggle('active', micActive);

    if (micActive) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach((track) => track.stop());
        } catch (error) {
            alert('Could not access microphone. Please check permissions.');
            micActive = false;
            micToggle.classList.remove('active');
        }
    }
}

function startVoiceInput() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('Speech recognition is not supported in this browser.');
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const oneShotRecognition = new SpeechRecognition();
    oneShotRecognition.continuous = false;
    oneShotRecognition.interimResults = false;
    oneShotRecognition.lang = 'en-US';
    oneShotRecognition.maxAlternatives = 1;
    voiceInputBtn.classList.add('active');

    oneShotRecognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        messageInput.value = transcript;
        autoResizeTextarea();
        voiceInputBtn.classList.remove('active');
    };

    oneShotRecognition.onerror = (event) => {
        const reason = event?.error || 'unknown';
        updateVoiceStatus(`Voice input error: ${reason}`);
        voiceInputBtn.classList.remove('active');
    };

    oneShotRecognition.onend = () => {
        voiceInputBtn.classList.remove('active');
    };

    oneShotRecognition.start();
}

function toggleVoiceSession() {
    if (voiceSessionBusy) {
        return;
    }

    if (voiceSessionActive) {
        stopAndSubmitVoiceSession();
    } else {
        startVoiceSession();
    }
}

function startVoiceSession() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('Speech recognition is not supported in this browser.');
        return;
    }

    if (voiceSessionBusy) {
        return;
    }

    voiceSessionActive = true;
    voiceSessionBusy = false;
    voiceStopRequested = false;
    voiceSessionBtn.disabled = false;
    voiceSessionBtn.textContent = 'Stop & Send';
    voiceSessionBtn.classList.add('active');
    voiceFinalBuffer = '';
    voiceInterimBuffer = '';
    clearVoiceSilenceTimer();
    setVoiceStageState('listening');
    updateVoiceStatus('Listening... click Stop & Send when finished.');
    beginVoiceCapture();
}

function stopAndSubmitVoiceSession() {
    if (!voiceSessionActive || voiceSessionBusy) {
        return;
    }

    voiceSessionBusy = true;
    voiceStopRequested = true;
    clearVoiceSilenceTimer();
    voiceSessionBtn.disabled = true;
    voiceSessionBtn.textContent = 'Processing...';
    updateVoiceStatus('Processing your speech...');

    if (voiceRecognition) {
        try {
            voiceRecognition.stop();
        } catch (error) {
            finalizeVoiceUtterance();
        }
    } else {
        finalizeVoiceUtterance();
    }
}

function stopVoiceSession() {
    voiceSessionActive = false;
    voiceSessionBusy = false;
    voiceStopRequested = false;
    voiceSessionBtn.textContent = 'Start Voice Chat';
    voiceSessionBtn.disabled = false;
    voiceSessionBtn.classList.remove('active');
    clearVoiceSilenceTimer();
    voiceFinalBuffer = '';
    voiceInterimBuffer = '';
    if (voiceRecognition) {
        voiceRecognition.onend = null;
        voiceRecognition.onresult = null;
        voiceRecognition.onerror = null;
        voiceRecognition.stop();
        voiceRecognition = null;
    }
    setVoiceStageState('idle');
    updateVoiceStatus('Ready. Press Start Voice Chat.');
}

function beginVoiceCapture() {
    if (!voiceSessionActive || voiceSessionBusy) {
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    voiceRecognition = new SpeechRecognition();
    voiceRecognition.continuous = true;
    voiceRecognition.interimResults = true;
    voiceRecognition.lang = 'en-US';
    voiceRecognition.maxAlternatives = 1;

    voiceRecognition.onresult = (event) => {
        if (!voiceSessionActive || voiceSessionBusy) return;

        let interim = '';
        for (let i = event.resultIndex; i < event.results.length; i += 1) {
            const result = event.results[i];
            const text = (result?.[0]?.transcript || '').trim();
            if (!text) continue;
            if (result.isFinal) {
                voiceFinalBuffer += `${voiceFinalBuffer ? ' ' : ''}${text}`;
            } else {
                interim += `${interim ? ' ' : ''}${text}`;
            }
        }

        voiceInterimBuffer = interim;

        const previewText = (voiceFinalBuffer || voiceInterimBuffer).trim();
        if (previewText) {
            updateVoiceStatus(`Listening: ${previewText}`);
        }
    };

    voiceRecognition.onerror = (event) => {
        if (voiceSessionActive) {
            setVoiceStageState('idle');
            const reason = event?.error || 'unknown';
            if (reason === 'not-allowed' || reason === 'service-not-allowed') {
                updateVoiceStatus('Microphone permission denied. Allow mic access in browser settings.');
                return;
            }
            if (reason === 'no-speech') {
                updateVoiceStatus('No speech detected. Listening again...');
            } else {
                updateVoiceStatus(`Microphone issue (${reason}). Retrying...`);
            }
            clearVoiceSilenceTimer();
            setTimeout(beginVoiceCapture, 900);
        }
    };

    voiceRecognition.onend = () => {
        if (voiceStopRequested && voiceSessionBusy) {
            finalizeVoiceUtterance();
            return;
        }

        if (voiceSessionActive && !voiceSessionBusy && !voiceStopRequested) {
            setTimeout(beginVoiceCapture, 250);
        }
    };

    try {
        voiceRecognition.start();
    } catch (error) {
        if (voiceSessionActive) {
            setTimeout(beginVoiceCapture, 600);
        }
    }
}

function clearVoiceSilenceTimer() {
    if (voiceSilenceTimer) {
        clearTimeout(voiceSilenceTimer);
        voiceSilenceTimer = null;
    }
}

function resetVoiceSessionButton() {
    voiceSessionActive = false;
    voiceSessionBusy = false;
    voiceStopRequested = false;
    voiceSessionBtn.disabled = false;
    voiceSessionBtn.textContent = 'Start Voice Chat';
    voiceSessionBtn.classList.remove('active');
}

async function finalizeVoiceUtterance() {
    const transcript = `${voiceFinalBuffer} ${voiceInterimBuffer}`.trim();

    if (voiceRecognition) {
        voiceRecognition.onend = null;
        voiceRecognition.onresult = null;
        voiceRecognition.onerror = null;
        voiceRecognition = null;
    }

    voiceFinalBuffer = '';
    voiceInterimBuffer = '';

    if (!transcript) {
        resetVoiceSessionButton();
        setVoiceStageState('idle');
        updateVoiceStatus('No speech captured. Press Start Voice Chat.');
        return;
    }

    appendMessage('user', transcript);

    await requestAssistantResponse(transcript, true);

    resetVoiceSessionButton();
    setVoiceStageState('idle');
    updateVoiceStatus('Ready. Press Start Voice Chat.');
}

function playAudio(url) {
    setVoiceStageState('speaking');
    ttsAudio.src = normalizeAudioUrl(url);
    ttsAudio.play().catch(() => {});
    ttsAudio.onended = () => {
        if (voiceSessionActive) {
            setVoiceStageState('listening');
        } else {
            setVoiceStageState('idle');
        }
    };
}

async function playAudioAndWait(url) {
    return new Promise((resolve) => {
        setVoiceStageState('speaking');
        ttsAudio.src = normalizeAudioUrl(url);
        const cleanUp = () => {
            ttsAudio.removeEventListener('ended', onEnded);
            ttsAudio.removeEventListener('error', onError);
            if (voiceSessionActive) {
                setVoiceStageState('listening');
            } else {
                setVoiceStageState('idle');
            }
            resolve();
        };
        const onEnded = () => cleanUp();
        const onError = () => cleanUp();
        ttsAudio.addEventListener('ended', onEnded, { once: true });
        ttsAudio.addEventListener('error', onError, { once: true });
        ttsAudio.play().catch(() => cleanUp());
    });
}

function normalizeAudioUrl(url) {
    if (!url) return '';
    if (url.startsWith('http://') || url.startsWith('https://')) return url;
    return `${API_BASE}${url}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

window.playAudio = playAudio;

function enableCameraDrag() {
    if (cameraDragEnabled) return;
    cameraDragEnabled = true;

    cameraPreview.addEventListener('pointerdown', (event) => {
        if (event.button !== 0) return;
        cameraPreview.setPointerCapture(event.pointerId);
        const rect = cameraPreview.getBoundingClientRect();
        cameraDragState = {
            pointerId: event.pointerId,
            offsetX: event.clientX - rect.left,
            offsetY: event.clientY - rect.top,
        };
        cameraPreview.classList.add('dragging');
    });

    cameraPreview.addEventListener('pointermove', (event) => {
        if (!cameraDragState || event.pointerId !== cameraDragState.pointerId) return;

        const maxX = Math.max(0, window.innerWidth - cameraPreview.offsetWidth - 8);
        const maxY = Math.max(0, window.innerHeight - cameraPreview.offsetHeight - 8);
        const nextLeft = Math.min(maxX, Math.max(8, event.clientX - cameraDragState.offsetX));
        const nextTop = Math.min(maxY, Math.max(72, event.clientY - cameraDragState.offsetY));

        cameraPreview.style.left = `${nextLeft}px`;
        cameraPreview.style.top = `${nextTop}px`;
        cameraPreview.style.right = 'auto';
        cameraPreview.style.bottom = 'auto';
    });

    const stopDrag = () => {
        if (!cameraDragState) return;
        cameraDragState = null;
        cameraPreview.classList.remove('dragging');
    };

    cameraPreview.addEventListener('pointerup', stopDrag);
    cameraPreview.addEventListener('pointercancel', stopDrag);
}
