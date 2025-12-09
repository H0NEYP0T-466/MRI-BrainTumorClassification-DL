/**
 * Preprocessing Steps Component
 * Displays intermediate preprocessing images in card format
 */

import React from 'react';
import './PreprocessingSteps.css';

interface PreprocessingStep {
  label: string;
  image: string;
  description: string;
}

interface PreprocessingStepsProps {
  steps: {
    original: string;
    denoised: string;
    contrast_enhanced: string;
    sharpened: string;
    edge_enhanced: string;
    normalized: string;
    segmented: string;
    final: string;
  };
}

const PreprocessingSteps: React.FC<PreprocessingStepsProps> = ({ steps }) => {
  const [imageErrors, setImageErrors] = React.useState<Set<number>>(new Set());

  const handleImageError = (index: number) => {
    setImageErrors(prev => new Set(prev).add(index));
  };

  const preprocessingSteps: PreprocessingStep[] = [
    {
      label: 'Original Image',
      image: steps.original,
      description: 'Uploaded MRI scan'
    },
    {
      label: 'After Denoising',
      image: steps.denoised,
      description: 'Non-Local Means denoising applied'
    },
    {
      label: 'Contrast Enhanced',
      image: steps.contrast_enhanced,
      description: 'CLAHE contrast enhancement'
    },
    {
      label: 'After Sharpening',
      image: steps.sharpened,
      description: 'Unsharp masking applied'
    },
    {
      label: 'Edge Enhanced',
      image: steps.edge_enhanced,
      description: 'Morphological gradient enhancement'
    },
    {
      label: 'Normalized',
      image: steps.normalized,
      description: 'Intensity normalization'
    },
    {
      label: 'Brain Segmented',
      image: steps.segmented,
      description: 'Skull stripping (brain extraction)'
    },
    {
      label: 'Final Image',
      image: steps.final,
      description: 'Preprocessed image for model input'
    }
  ];

  return (
    <div className="preprocessing-container">
      <h3 className="preprocessing-title">Image Preprocessing Pipeline</h3>
      <p className="preprocessing-subtitle">
        Step-by-step transformations applied to the uploaded MRI image
      </p>
      
      <div className="preprocessing-grid">
        {preprocessingSteps.map((step, index) => (
          <div key={index} className="preprocessing-card">
            <div className="preprocessing-card-header">
              <span className="step-number">{index + 1}</span>
              <h4 className="step-label">{step.label}</h4>
            </div>
            
            <div className="preprocessing-card-image">
              {imageErrors.has(index) ? (
                <div className="image-error">
                  <p>Failed to load image</p>
                </div>
              ) : (
                <img 
                  src={step.image} 
                  alt={step.label}
                  className="step-image"
                  onError={() => handleImageError(index)}
                />
              )}
            </div>
            
            <div className="preprocessing-card-footer">
              <p className="step-description">{step.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PreprocessingSteps;
