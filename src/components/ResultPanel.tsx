import React from 'react';
import './ResultPanel.css';
interface ResultPanelProps {
  className: string;
  confidence: number;
}

const ResultPanel: React.FC<ResultPanelProps> = ({ className, confidence }) => {
  const isTumor = className === 'tumor';
  const displayClass = isTumor ? 'Tumor Detected' : 'No Tumor';
  const confidencePercent = (confidence * 100).toFixed(2);
  
  return (
    <div className="result-panel">
      <div className={`result-header ${isTumor ? 'tumor' : 'no-tumor'}`}>
        <h3 className="result-title">Prediction Result</h3>
      </div>
      
      <div className="result-content">
        <div className="result-class">
          <span className="result-label">Classification:</span>
          <span className={`result-value ${isTumor ? 'tumor' : 'no-tumor'}`}>
            {displayClass}
          </span>
        </div>
        
        <div className="result-confidence">
          <span className="result-label">Confidence:</span>
          <div className="confidence-container">
            <div className="confidence-bar-bg">
              <div 
                className={`confidence-bar ${isTumor ? 'tumor' : 'no-tumor'}`}
                style={{ width: `${confidencePercent}%` }}
              ></div>
            </div>
            <span className="confidence-value">{confidencePercent}%</span>
          </div>
        </div>
        
      </div>
    </div>
  );
};

export default ResultPanel;
