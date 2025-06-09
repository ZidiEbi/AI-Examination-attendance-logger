// Initialize camera
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const resultDiv = document.getElementById("result");

// Check for camera support
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
      startRecognition();
    })
    .catch((err) => {
      resultDiv.textContent = `Camera Error: ${err.message}`;
      resultDiv.style.display = "block";
    });
} else {
  resultDiv.textContent = "Camera API not supported in this browser";
  resultDiv.style.display = "block";
}

// Recognition function
function startRecognition() {
  setInterval(() => {
    canvas.getContext("2d").drawImage(video, 0, 0, 640, 480);
    canvas.toBlob((blob) => {
      const formData = new FormData();
      formData.append("image", blob, "frame.jpg");
      formData.append("exam_id", document.getElementById("exam_id").value); // Add exam ID

      fetch("/verify", {
        // Updated endpoint to match alias
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.status === "success") {
            resultDiv.textContent = `Verified: ${data.student_id}`;
            resultDiv.className = "alert alert-success";
          } else if (data.status === "rejected") {
            resultDiv.textContent = `Rejected: ${data.reason}`;
            resultDiv.className = "alert alert-warning";
          } else {
            resultDiv.textContent =
              "Error: " + (data.message || "Unknown error");
            resultDiv.className = "alert alert-danger";
          }
          resultDiv.style.display = "block";
        })
        .catch((err) => {
          resultDiv.textContent = `API Error: ${err.message}`;
          resultDiv.className = "alert alert-danger";
          resultDiv.style.display = "block";
        });
    }, "image/jpeg");
  }, 5000); // Process every 5 seconds
}
