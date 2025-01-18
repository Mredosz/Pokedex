const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { spawn } = require("child_process");
const path = require("path");

const app = express();

app.use(express.json());

app.use(cors({ origin: "http://localhost:5173" }));

const upload = multer({ dest: "uploads/" });

app.get("/metrics", (req, res) => {
  const pythonProcess = spawn("python", ["../src/Metrics.py"]);

  let result = '';
  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.on('close', () => {
    res.json(JSON.parse(result.trim()));
  });

  pythonProcess.stderr.on('data', (error) => {
    console.error(`Error: ${error}`);
    res.status(500).send('Error during prediction');
  });
});



app.post("/predict", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).send("No image file uploaded.");
  }

  const imagePath = path.resolve(req.file.path);

  const pythonProcess = spawn("python", ["../src/PredictImageForEachModel.py", imagePath]);

  let result = '';
  pythonProcess.stdout.on('data', (data) => {
    result += data.toString();
  });

  pythonProcess.on('close', () => {
    res.json(JSON.parse(result.trim()));
  });

  pythonProcess.stderr.on('data', (error) => {
    console.error(`Error: ${error}`);
    res.status(500).send('Error during prediction');
  });
});


app.listen(3000, () => {
  console.log("Server listening at port 3000");
});