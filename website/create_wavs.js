const fs = require('fs');

const PATH = "samples/";
const result = []

fs.readdir(PATH, (err, files) => {
    files.forEach(file => {
        result.push(PATH + file);
    });

    fs.writeFileSync('./wavs.json', JSON.stringify(result));
});
