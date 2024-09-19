import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [file, setFile] = useState(null);
    const [query, setQuery] = useState('');
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);  // Loading state

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            setLoading(true);  // Start loading
            const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            alert('File uploaded and processed successfully');
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('Error uploading file: ' + (error.response?.data?.error || error.message));
        } finally {
            setLoading(false);  // Stop loading
        }
    };

    const handleQuery = async () => {
        if (!query) return;

        try {
            setLoading(true);  // Start loading
            const response = await axios.post('http://127.0.0.1:5000/query', { query });
            setResponse(response.data);
        } catch (error) {
            console.error('Error querying PDF:', error);
            alert('Error querying PDF');
        } finally {
            setLoading(false);  // Stop loading
        }
    };

    return (
        <div className="App">
            <h1>PDF Query Chatbot</h1>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload} disabled={loading}>
                {loading ? 'Uploading...' : 'Upload PDF'}
            </button>
            <br />
            <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question"
                disabled={loading}
            />
            <button onClick={handleQuery} disabled={loading}>
                {loading ? 'Submitting...' : 'Submit Query'}
            </button>

            {loading && <p>Loading...</p>}  {/* Loading message */}

            <div>
                {response && (
                    <div>
                        <h2>Answer:</h2>
                        <p>{response.answer}</p>
                    
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
