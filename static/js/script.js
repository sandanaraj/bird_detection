let birdDetected = false;


setInterval(() => {
    fetch('/bird_status')
        .then(response => response.json())
        .then(data => {
            if (data.bird_detected && !birdDetected) {
                birdDetected = true;
                document.getElementById('alert').classList.add('detected');
            } else if (!data.bird_detected && birdDetected) {
                birdDetected = false;
                document.getElementById('alert').classList.remove('detected');
            }
        });
}, 500); 