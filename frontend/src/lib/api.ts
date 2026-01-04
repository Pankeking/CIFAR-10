export interface BaseResponse {
  status: string;
  message: string;
}
export interface ModelInfo {
  filename: string;
  dataset: string;
  loss: string;
  samples: number;
  epochs: number;
  ext: string;
}

export interface ImageInfo {
    image: string;
    true_label: number;
}

export interface PredictInfo {
    pred_label: string;
    true_label: string;
    probs: Record<string, number>;
}

export interface HealthInfo {
  torch_device: string;
}

export interface HealthResponse extends BaseResponse {
  health: HealthInfo;
}

export interface ModelListResponse extends BaseResponse {
  models: ModelInfo[];
}

export interface ImageResponse extends BaseResponse {
    image: ImageInfo;
}

export interface PredictResponse extends BaseResponse {
    prediction: PredictInfo;
}

export async function apiFetch<T>(endpoint: string): Promise<T> {
  const res = await fetch(`/api${endpoint}`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  console.log('API response received from server')
  const data = await res.json()
  if (data.status !== 'success') throw new Error(`API error: ${data.message}`)
  console.log('API data:', data)
  return data as T
}


  