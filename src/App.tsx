/**
 * Main Application Component
 * MRI Brain Tumor Classification - Single Page Application
 */

import { useState, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultPanel from './components/ResultPanel';
import ProgressBar from './components/ProgressBar';
import PreprocessingSteps from './components/PreprocessingSteps';
import { predictImage, checkHealth } from './api/client';
import type { PredictionResponse } from './api/client';
import styles from './styles/app.module.css';
import './styles/variables.css';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);

  // Check server health on mount
  useEffect(() => {
    const checkServerHealth = async () => {
      try {
        const health = await checkHealth();
        setModelLoaded(health.model_loaded);
      } catch (err) {
        console.error('Failed to check server health:', err);
        setModelLoaded(false);
      }
    };

    checkServerHealth();
  }, []);

  const handleUpload = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await predictImage(file);
      setResult(prediction);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to analyze image';
      setError(errorMessage);
      console.error('Prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className={styles.app}>
      <div className={styles.container}>
        {/* Header */}
        <header className={styles.header}>
          <h1 className={styles.title}>MRI Brain Tumor Classification</h1>
          <p className={styles.subtitle}>
            Deep Learning-based Brain Tumor Detection System
          </p>
          {modelLoaded !== null && (
            <div style={{ marginTop: '1rem' }}>
              <span
                style={{
                  color: modelLoaded ? 'var(--accent-success)' : 'var(--accent-warning)',
                  fontSize: '0.9rem',
                }}
              >
                {modelLoaded ? '✓ Model Ready' : '⚠ Model Not Loaded'}
              </span>
            </div>
          )}
        </header>

        {/* Main Content */}
        <div className={styles.mainContent}>
          {/* Upload Section */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Upload MRI Image</h2>
            <ImageUpload onUpload={handleUpload} isLoading={isLoading} onReset={handleReset} />
          </section>

          {/* Results Section */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Analysis Results</h2>
            
            {isLoading && (
              <ProgressBar message="Analyzing MRI scan..." />
            )}

            {error && !isLoading && (
              <div className={styles.error}>
                <p className={styles.errorTitle}>Error</p>
                <p className={styles.errorMessage}>{error}</p>
              </div>
            )}

            {result && !isLoading && (
              <>
                <ResultPanel
                  className={result.class}
                  confidence={result.confidence}
                />
                
                {result.preprocessing_steps && (
                  <PreprocessingSteps steps={result.preprocessing_steps} />
                )}
              </>
            )}

            {!isLoading && !error && !result && (
              <div className={styles.loading}>
                <p className={styles.loadingText}>
                  Upload an MRI image to begin analysis
                </p>
              </div>
            )}
          </section>
        </div>

        {/* Footer */}
        <footer className={styles.footer}>
          <p>
            MRI Brain Tumor Classification System © 2024 | 
            Powered by Vision Transformer (ViT) Deep Learning Architecture
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
