let ws = null;
let sessionId = null;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let analyserNode = null;
let levelAnimFrame = null;
let timerInterval = null;
let recordingSeconds = 0;
let isRecording = false;

// ── WebSocket ──

function autoConnect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = protocol + '//' + window.location.host + '/ws/stream';

    setConnBadge('connecting', 'Connecting');
    ws = new WebSocket(url);

    ws.onopen = function() {
        setConnBadge('connected', 'Connected');
        document.getElementById('recordBtn').disabled = false;
    };

    ws.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleMessage(data);
        } catch (e) {
            addMessage('error', 'Failed to parse message: ' + event.data);
        }
    };

    ws.onerror = function(error) {
        addMessage('error', 'WebSocket error occurred');
        console.error('WebSocket error:', error);
    };

    ws.onclose = function() {
        setConnBadge('disconnected', 'Disconnected');
        sessionId = null;
        stopRecording();
        document.getElementById('recordBtn').disabled = true;
    };
}

function setConnBadge(state, label) {
    const badge = document.getElementById('connBadge');
    badge.className = 'conn-badge ' + state;
    badge.textContent = label;
}

// ── Recording ──

async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

        audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(mediaStream);

        // 1024-sample buffer ≈ 21ms at 48kHz native → ~20ms at 16kHz after resample
        scriptProcessor = audioContext.createScriptProcessor(1024, 1, 1);
        const targetSampleRate = 16000;
        const nativeSampleRate = audioContext.sampleRate;

        scriptProcessor.onaudioprocess = function(e) {
            if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
            const float32 = e.inputBuffer.getChannelData(0);
            const pcm = resampleAndConvert(float32, nativeSampleRate, targetSampleRate);
            ws.send(JSON.stringify({
                type: 'audio',
                data: int16ToBase64(pcm),
                sample_rate: targetSampleRate
            }));
        };

        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 256;

        source.connect(analyserNode);
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        isRecording = true;
        startTimer();
        updateLevelMeter();
        setRecordingUI(true);
    } catch (e) {
        addMessage('error', 'Microphone error: ' + e.message);
    }
}

function stopRecording() {
    isRecording = false;

    if (levelAnimFrame)  { cancelAnimationFrame(levelAnimFrame); levelAnimFrame = null; }
    if (timerInterval)   { clearInterval(timerInterval); timerInterval = null; }
    if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
    if (analyserNode)    { analyserNode.disconnect(); analyserNode = null; }
    if (mediaStream)     { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (audioContext)    { audioContext.close(); audioContext = null; }

    document.getElementById('levelBar').style.width = '0%';
    document.getElementById('recordTimer').textContent = '00:00';
    setRecordingUI(false);
}

// ── Timer ──

function startTimer() {
    recordingSeconds = 0;
    updateTimerDisplay();
    timerInterval = setInterval(() => { recordingSeconds++; updateTimerDisplay(); }, 1000);
}

function updateTimerDisplay() {
    const m = String(Math.floor(recordingSeconds / 60)).padStart(2, '0');
    const s = String(recordingSeconds % 60).padStart(2, '0');
    document.getElementById('recordTimer').textContent = `${m}:${s}`;
}

// ── Level meter ──

function updateLevelMeter() {
    if (!analyserNode) return;
    const data = new Uint8Array(analyserNode.frequencyBinCount);
    analyserNode.getByteFrequencyData(data);
    const avg = data.reduce((a, b) => a + b, 0) / data.length;
    document.getElementById('levelBar').style.width = Math.min(100, (avg / 80) * 100) + '%';
    levelAnimFrame = requestAnimationFrame(updateLevelMeter);
}

// ── UI state ──

function setRecordingUI(recording) {
    const btn = document.getElementById('recordBtn');
    const status = document.getElementById('recordStatus');

    btn.querySelector('.icon-mic').hidden  = recording;
    btn.querySelector('.icon-stop').hidden = !recording;
    btn.classList.toggle('recording', recording);

    status.textContent = recording ? 'Recording…' : 'Ready';
    status.className = 'status-text' + (recording ? ' recording' : '');
}

// ── Audio helpers ──

function resampleAndConvert(float32, fromRate, toRate) {
    const ratio = fromRate / toRate;
    const outLength = Math.round(float32.length / ratio);
    const int16 = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
        const srcIdx = Math.min(Math.round(i * ratio), float32.length - 1);
        const s = Math.max(-1, Math.min(1, float32[srcIdx]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
}

function int16ToBase64(int16Array) {
    const uint8 = new Uint8Array(int16Array.buffer);
    let binary = '';
    const chunkSize = 0x8000;
    for (let i = 0; i < uint8.length; i += chunkSize) {
        binary += String.fromCharCode(...uint8.subarray(i, i + chunkSize));
    }
    return btoa(binary);
}

// ── Message handling ──

function handleMessage(data) {
    switch (data.type) {
        case 'transcript':
            addMessage('transcript', data.text, data.is_final);
            break;
        case 'error':
            addMessage('error', data.message);
            break;
        case 'session_info':
            sessionId = data.session_id;
            addMessage('session-info', `Session: ${data.session_id} · ${data.status}`);
            break;
        default:
            addMessage('system', 'Unknown message type: ' + data.type);
    }
}

function addMessage(type, content, isFinal = false) {
    const messagesDiv = document.getElementById('messages');

    const emptyHint = document.getElementById('emptyHint');
    if (emptyHint) emptyHint.remove();

    const messageDiv = document.createElement('div');
    const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    let className = 'message';
    let typeLabel;

    switch (type) {
        case 'transcript':
            className += isFinal ? ' transcript final' : ' transcript';
            typeLabel = isFinal ? 'Final' : 'Partial';
            break;
        case 'error':
            className += ' error';
            typeLabel = 'Error';
            break;
        case 'session-info':
            className += ' session-info';
            typeLabel = 'Session';
            break;
        default:
            className += ' system';
            typeLabel = 'System';
    }

    messageDiv.className = className;
    messageDiv.innerHTML = `
        <div class="msg-meta">
            <span class="msg-type">${typeLabel}</span>
            <span class="msg-time">${timestamp}</span>
        </div>
        <div class="msg-content">${content}</div>
    `;

    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function clearMessages() {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = '<p class="empty-hint" id="emptyHint">Transcripts will appear here…</p>';
}

// ── Init ──

window.addEventListener('load', autoConnect);

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        if (!document.getElementById('recordBtn').disabled) toggleRecording();
    }
});
