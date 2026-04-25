const socket = io();
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('overlay');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const loading = document.getElementById('loading');
const mainPred = document.getElementById('mainPred');
const modelTable = document.getElementById('modelTable');

let stream = null;
let isRunning = false;

startBtn.onclick = async () => {
    try {
        loading.classList.remove('hidden');
        startBtn.disabled = true;
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        await video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        stopBtn.disabled = false;
        isRunning = true;
        loading.classList.add('hidden');
        requestAnimationFrame(processFrame);
    } catch (err) {
        alert("Camera access denied: " + err.message);
        loading.classList.add('hidden');
        startBtn.disabled = false;
    }
};

stopBtn.onclick = () => {
    stream?.getTracks().forEach(t => t.stop());
    isRunning = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    overlay.src = "";
    mainPred.textContent = "Stopped";
    modelTable.innerHTML = "";
};

async function processFrame() {
    if (!isRunning) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onloadend = () => socket.emit('video_frame', reader.result);
        reader.readAsDataURL(blob);
    }, 'image/jpeg', 0.6);
    setTimeout(() => requestAnimationFrame(processFrame), 150); // ~6 FPS
}

socket.on('prediction_result', (data) => {
    overlay.src = data.frame;
    mainPred.textContent = `${data.label} (${Math.round(data.confidence * 100)}%)`;
    mainPred.style.color = data.label === "Mask" ? "#22c55e" : "#ef4444";
    
    modelTable.innerHTML = Object.entries(data.details).map(([name, info]) => `
        <div class="model-card">
            <div class="model-name">${name}</div>
            <div class="model-val">${info.label}</div>
            <div class="conf">${Math.round(info.confidence * 100)}% conf</div>
        </div>
    `).join('');
});

socket.on('status', d => mainPred.textContent = d.msg);
socket.on('error', d => mainPred.textContent = "❌ " + d.msg);