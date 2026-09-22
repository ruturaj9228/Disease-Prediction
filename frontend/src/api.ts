import axios from 'axios';

const API_URL = 'http://127.0.0.1:8002';

export interface PredictionResult {
  disease: string;
  confidence: number;
  contributing_symptoms: string[];
}

export const fetchSymptoms = async (): Promise<string[]> => {
  const { data } = await axios.get(`${API_URL}/symptoms`);
  return data;
};

export const predictDisease = async (symptoms: string[]): Promise<PredictionResult[]> => {
  const { data } = await axios.post(`${API_URL}/predict`, { symptoms });
  return data;
};
