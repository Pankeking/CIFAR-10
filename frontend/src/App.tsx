'use client'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { ModelViewer } from './components/ModelViewer'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { apiFetch } from './lib/api'  // Utils below
import type { ModelListResponse, HealthResponse } from './lib/api'

export default function App() {
  const [selectedModel, setSelectedModel] = useState<string>('')
  const healthQuery = useQuery<HealthResponse>({ queryKey: ['health'], queryFn: () => apiFetch('/health') })
  const modelsQuery = useQuery({ queryKey: ['models'], queryFn: () => apiFetch<ModelListResponse>('/models') })

  if (healthQuery.isLoading || modelsQuery.isLoading) return <div>Loading...</div>

  return (
    <div className="container mx-auto p-8 max-w-4xl">
      <Card>
        <CardHeader>
          <CardTitle>ML Model Viewer</CardTitle>
          <Badge variant="secondary">{healthQuery.data?.health.torch_device ?? 'CPU'}</Badge>
        </CardHeader>
        <CardContent className="space-y-4">
          <select
            className="w-full p-2 border rounded"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            <option value="">Select Model</option>
            {modelsQuery.data?.models.map((m) => (
              <option key={m.filename} value={m.filename}>
                {m.dataset} | {m.samples.toLocaleString()} samples | {m.epochs} epochs
              </option>
            ))}
          </select>
          {selectedModel && <ModelViewer modelFilename={selectedModel} />}
        </CardContent>
      </Card>
    </div>
  )
}
