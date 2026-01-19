const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileElem');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const originalImg = document.getElementById('original-img');
const processedImg = document.getElementById('processed-img');
const craterCount = document.getElementById('crater-count');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
});

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, unhighlight, false);
});

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    dropArea.classList.add('highlight');
}

function unhighlight(e) {
    dropArea.classList.remove('highlight');
}

function handleDrop(e) {
    var dt = e.dataTransfer;
    var files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function uploadFile(file) {
    // UI Updates
    results.classList.add('hidden');
    loading.classList.remove('hidden');

    let url = '/detect';
    let formData = new FormData();
    formData.append('image', file);

    fetch(url, {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            loading.classList.add('hidden');
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            displayResults(data);
        })
        .catch(() => {
            loading.classList.add('hidden');
            alert("An error occurred during upload or processing.");
        });
}

function displayResults(data) {
    // Add cache buster to images just in case
    originalImg.src = data.original_url;
    processedImg.src = data.processed_url;

    craterCount.innerText = data.count;

    // Animation for numbers
    animateValue(craterCount, 0, data.count, 1000);

    results.classList.remove('hidden');
    results.scrollIntoView({ behavior: 'smooth' });
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
