import React, { useState, useEffect } from 'react';
import axios from 'axios';
import D3Visualization from './D3Visualization';
const json_path = './image_matrix.json';


const UploadCSV = () => {
  const [file, setFile] = useState(null);
  const [option, setOption] = useState('1');


  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleOptionChange = (e) => {
    setOption(e.target.value);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('option', option);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log('File uploaded successfully');
    } catch (error) {
      console.error('Error uploading file: ', error);
    }
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <select value={option} onChange={handleOptionChange}>
        <option value="1">Top KNN</option>
        <option value="2">Top Spiral</option>
        <option value="3">Single Dim KNN</option>
        <option value="4">Limit</option>
        <option value="5">New</option>
        <option value="6">Individual</option>
        <option value="7">Other</option>
      </select>
      <button onClick={handleUpload}>Upload</button>

      {/* Render the output data as a D3 visualization */}
      <div>
        <h2>Processed Data Visualization</h2>
        <D3Visualization filepath={json_path} />
      </div>
    </div>
  );
};

export default UploadCSV;
