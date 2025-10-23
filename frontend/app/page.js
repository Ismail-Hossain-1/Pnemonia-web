"use client"

import { useState, useRef } from 'react';
import Head from 'next/head';

export default function ImageClassifier() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  // Update these with your actual class names
  const classNames = {
    0: 'Class 0', // e.g., 'Normal', 'Cat', etc.
    1: 'Class 1'  // e.g., 'Pneumonia', 'Dog', etc.
  };

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }

      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError('');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError('');
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64data = reader.result;

        const response = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image: base64data,
          }),
        });

        const data = await response.json();

        if (data.success) {
          setPrediction(data);
        } else {
          setError(data.error || 'Prediction failed');
        }
        setLoading(false);
      };

      reader.readAsDataURL(selectedImage);
    } catch (err) {
      setError('Failed to connect to the server. Make sure the backend is running on port 5000.');
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setPreviewUrl('');
    setPrediction(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBgColor = (confidence) => {
    if (confidence >= 0.8) return 'bg-green-50 border-green-200';
    if (confidence >= 0.6) return 'bg-yellow-50 border-yellow-200';
    return 'bg-red-50 border-red-200';
  };

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <Head>
        <title>PNEUMONIA Image Classifier</title>
        <meta name="description" content="Test your binary image classification model" />
      </Head>

      <div className="max-w-4xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            PNEUMONIA Image Classification
          </h1>
          <p className="text-gray-600">
            Using Sigmoid Activation · Binary Crossentropy
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Upload Image</h2>
            
            <div
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 transition-colors"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleImageSelect}
                accept="image/*"
                className="hidden"
              />
              
              {previewUrl ? (
                <div className="space-y-4">
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="max-h-64 mx-auto rounded-lg object-contain"
                  />
                  <p className="text-sm text-gray-600">Click or drag to change image</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="text-4xl text-gray-400">📁</div>
                  <p className="text-gray-600">Click or drag and drop an image here</p>
                  <p className="text-sm text-gray-500">Supports JPG, PNG, JPEG</p>
                </div>
              )}
            </div>

            <div className="flex space-x-4 mt-6">
              <button
                onClick={handlePredict}
                disabled={!selectedImage || loading}
                className="flex-1 bg-blue-500 text-white py-3 px-4 rounded-lg hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-semibold"
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Predicting...
                  </span>
                ) : (
                  'Predict'
                )}
              </button>
              
              <button
                onClick={handleReset}
                className="flex-1 bg-gray-500 text-white py-3 px-4 rounded-lg hover:bg-gray-600 transition-colors font-semibold"
              >
                Reset
              </button>
            </div>
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4">Prediction Results</h2>
            
            {error && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                {error}
              </div>
            )}

            {prediction ? (
              <div className="space-y-6">
                <div className={`border rounded-lg p-4 ${getConfidenceBgColor(prediction.confidence)}`}>
                  <h3 className="font-semibold text-gray-800 mb-2">Prediction</h3>
                  <p className="text-2xl font-bold text-gray-900">
                    {prediction.prediction}
                  </p>
                  <p className={`text-lg font-semibold mt-2 ${getConfidenceColor(prediction.confidence)}`}>
                    Confidence: {(prediction.confidence * 100).toFixed(2)}%
                  </p>
                  <p className="text-sm text-gray-600 mt-1">
                    Raw probability: {(prediction.prediction_probability * 100).toFixed(2)}%
                  </p>
                </div>

                <div>
                  <h4 className="font-semibold mb-3 text-gray-800">Class Probabilities</h4>
                  <div className="space-y-3">
                    {Object.entries(prediction.class_probabilities).map(([className, prob]) => (
                      <div key={className} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-gray-600">{className}</span>
                          <span className="font-medium">{(prob * 100).toFixed(2)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div
                            className="bg-blue-500 h-3 rounded-full transition-all duration-500"
                            style={{ width: `${prob * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-800 mb-2">Model Info</h4>
                  <p className="text-sm text-blue-700">
                    This is a binary classifier using sigmoid activation. 
                    The output represents the probability of {classNames[1]}.
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-12">
                <div className="text-4xl mb-4">🔍</div>
                <p className="text-lg mb-2">No prediction yet</p>
                <p className="text-sm">Upload an image and click Predict to see results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}