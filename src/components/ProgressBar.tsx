/**
 * Progress Bar Component
 * Displays animated loading indicator during prediction
 */

import React from 'react';
import './ProgressBar.css';

interface ProgressBarProps {
  message?: string;
}

const ProgressBar: React.FC<ProgressBarProps> = ({ message = 'Processing...' }) => {
  return (
    <div className="progress-container">
      <div className="spinner"></div>
      <p className="progress-message">{message}</p>
    </div>
  );
};

export default ProgressBar;
