// function captureAndDetect() {
//     fetch("/detect")
//     .then(response => response.json())
//     .then(data => {
//         if (data.error) {
//             alert("Error capturing image.");
//         } else {
//             document.getElementById("inputImage").src = data.input_image;
//             document.getElementById("outputImage").src = data.output_image;
//             document.getElementById("responseText").innerText = data.response;
//         }
//     })
//     .catch(error => console.error("Error:", error));
// }
