/**
 * Utility functions for normalizing and formatting labels
 */

export const normalizeLabel = (label: string): string => {
  return label.replace(/_/g, ' ').replace(/\b\w/, l => l.toUpperCase());
};
