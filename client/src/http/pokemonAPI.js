import axios from "axios";

export async function fetchPredictionData(img) {
    return (await axios.post("http://localhost:3000/predict", img)).data
}

export async function fetchMetricsData() {
    return (await axios.get("http://localhost:3000/metrics")).data
}