/**
 * Theme configuration for the application
 * Dark aesthetic with contrasting colors
 */

export const theme = {
  colors: {
    // Background colors
    background: {
      primary: '#111',
      secondary: '#1a1a1a',
      tertiary: '#252525',
      hover: '#2a2a2a',
    },
    // Text colors
    text: {
      primary: '#e0e0e0',
      secondary: '#b0b0b0',
      muted: '#808080',
    },
    // Accent colors
    accent: {
      danger: '#ff6b6b',
      info: '#4dd0e1',
      success: '#00e676',
      warning: '#ffd93d',
    },
    // Borders
    border: {
      default: '#333',
      light: '#444',
    },
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
  },
  borderRadius: {
    sm: '4px',
    md: '8px',
    lg: '12px',
    xl: '16px',
    full: '9999px',
  },
  shadows: {
    sm: '0 2px 4px rgba(0, 0, 0, 0.5)',
    md: '0 4px 8px rgba(0, 0, 0, 0.6)',
    lg: '0 8px 16px rgba(0, 0, 0, 0.7)',
  },
  transitions: {
    fast: '0.15s ease',
    base: '0.3s ease',
    slow: '0.5s ease',
  },
} as const;

export type Theme = typeof theme;
