import {fetchPredictionData, fetchMetricsData} from "../http/pokemonAPI.js"
import {useState} from "react";
import {useMutation, useQuery, useQueryClient} from "@tanstack/react-query";

export default function MainPage() {
    const [image, setImage] = useState(null);
    const [formData, setFormData] = useState(null);

    const mutation = useMutation({
        mutationKey: "prediction",
        mutationFn: (formData) => fetchPredictionData(formData),
    });


    const {data: metricsData, isLoading: metricsLoading, error: metricsError} = useQuery(
        {
            queryKey: ["metrics"], queryFn: fetchMetricsData,
        }
    );

    const handleImageChange = (event) => {
        setImage(event.target.files[0]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();

        if (!image) {
            alert("Please upload an image of a pokemon.");
            return;
        }

        const formData = new FormData();
        formData.append("image", image);

        setFormData(formData);
        await mutation.mutateAsync(formData);
    };

return (
  <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-4">
    <h1 className="text-4xl font-bold text-white mb-8">Pokédex</h1>
    <form
      onSubmit={handleSubmit}
      className="bg-gray-800 shadow-lg rounded-lg p-6 w-full max-w-md"
    >
      <div className="mb-4">
        <label
          htmlFor="image"
          className="block text-white font-medium mb-2"
        >
          Upload an image of a Pokémon:
        </label>
        <input
          type="file"
          id="image"
          accept="image/*"
          onChange={handleImageChange}
          className="block w-full text-white border border-gray-600 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 p-2 bg-gray-700"
        />
      </div>
      <button
        type="submit"
        className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-blue-500 transition duration-200"
      >
        Submit
      </button>
    </form>

    {mutation.isLoading && (
      <p className="text-blue-400 font-medium mt-4">Loading prediction...</p>
    )}
    {mutation.isError && (
      <p className="text-red-400 font-medium mt-4">
        Error: {mutation.error.message}
      </p>
    )}
    {mutation.data && (
  <div className="bg-gray-800 shadow-lg rounded-lg p-6 mt-6 w-full max-w-md">
    <h2 className="text-2xl font-bold text-white mb-4">Prediction Result</h2>
    <table className="w-full text-white text-left border-collapse">
      <thead>
        <tr className="border-b border-gray-600">
          <th className="py-2 px-4">Model</th>
          <th className="py-2 px-4">Prediction</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(mutation.data).map(([model, result], index) => (
          <tr
            key={model}
            className={index % 2 === 0 ? "bg-gray-700" : "bg-gray-800"}
          >
            <td className="py-2 px-4 font-medium">{model}</td>
            <td className="py-2 px-4">{result}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)}

    {metricsLoading && (
      <p className="text-blue-400 font-medium mt-4">Loading metrics...</p>
    )}
    {metricsError && (
      <p className="text-red-400 font-medium mt-4">Error fetching metrics</p>
    )}
    {metricsData && (
  <div className="bg-gray-800 shadow-lg rounded-lg p-6 mt-6 w-full max-w-lg">
    <h2 className="text-2xl font-bold text-white mb-4">Metrics Data</h2>
    <table className="w-full text-white text-left border-collapse">
      <thead>
        <tr className="border-b border-gray-600">
          <th className="py-2 px-4">Model</th>
          <th className="py-2 px-4">Accuracy</th>
          <th className="py-2 px-4">Precision</th>
          <th className="py-2 px-4">Recall</th>
          <th className="py-2 px-4">F1-Score</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(metricsData).map(([model, metrics], index) => (
          <tr
            key={model}
            className={index % 2 === 0 ? "bg-gray-700" : "bg-gray-800"}
          >
            <td className="py-2 px-4 font-medium">{model.replace(/_/g, " ")}</td>
            <td className="py-2 px-4">{metrics.Accuracy}</td>
            <td className="py-2 px-4">{metrics.Precision}</td>
            <td className="py-2 px-4">{metrics.Recall}</td>
            <td className="py-2 px-4">{metrics["F1-score"]}</td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)}

  </div>
);


}