/**
 * API client for backend communication
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface PredictionResponse {
  class: string;
  confidence: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface TrainResponse {
  epochs: number;
  final_metrics: {
    val_acc: number;
    train_acc: number;
    best_epoch: number;
    training_time_minutes: number;
  };
  model_path: string;
}

/**
 * Check server health and model status
 */
export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error('Failed to check server health');
  }
  
  return response.json();
}

/**
 * Predict tumor classification from uploaded image
 */
export async function predictImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || 'Prediction failed');
  }
  
  return response.json();
}

/**
 * Train the model
 */
export async function trainModel(epochs: number = 10, batchSize: number = 32): Promise<TrainResponse> {
  const response = await fetch(`${API_BASE_URL}/train`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      epochs,
      batch_size: batchSize,
    }),
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || 'Training failed');
  }
  
  return response.json();
}
