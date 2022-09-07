let WAVS = []

function loadNextWav() {
    const next_index = Math.floor(Math.random() * WAVS.length);
    const ni = WAVS[next_index];
    const nel = `<audio controls="controls"><source src="${ni}" type="audio/x-wav" /></audio>`;
    console.log('Next element: ', next_index, ni, nel);
    document.getElementById('player').innerHTML = nel;
}

fetch("./wavs.json").then(response => {
    console.log(response);
    return response.json()
}).then(json => {
    WAVS = json;
    console.log(WAVS);
}).then(loadNextWav);
