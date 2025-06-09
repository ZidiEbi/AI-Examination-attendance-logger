// Access camera
function startCamera() {
    const video = document.getElementById('video');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => video.srcObject = stream)
        .catch(err => console.error("Camera error:", err));

    // Capture frame every 5 seconds
    setInterval(captureAndRecognize, 5000);
}

// Send frame to Flask API
function captureAndRecognize() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 640, 480);

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');

        fetch('/api/recognize', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'recognized') {
                alert(`Welcome, ${data.name}! Attendance marked.`);
                window.location.reload();  // Refresh dashboard
            } else {
                alert("Unknown student. Please report to the department.");
            }
        });
    }, 'image/jpeg', 0.9);
}