'use client'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'  // Add import
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Card, CardContent } from '@/components/ui/card'
import { apiFetch, type ImageResponse, type PredictResponse } from '../lib/api'  // Typed fetches

interface Props { modelFilename: string }

export function ModelViewer({ modelFilename }: Props) {
  const [idx, setIdx] = useState(0)
  
  const imageQuery = useQuery<ImageResponse>({
    queryKey: ['image', modelFilename, idx],
    queryFn: () => apiFetch<ImageResponse>(`/image/${idx}?model_filename=${modelFilename}`),
    enabled: !!modelFilename,
  })
  const predQuery = useQuery<PredictResponse>({
    queryKey: ['predict', modelFilename, idx],
    queryFn: () => apiFetch<PredictResponse>(`/predict/${idx}?model_filename=${modelFilename}`),
    enabled: !!modelFilename,
  })

  const next = () => setIdx((i) => Math.min(i + 1, 9999))
  const prev = () => setIdx((i) => Math.max(i - 1, 0))

  if (imageQuery.isPending) return <div className="text-center">Loading image...</div>
  if (imageQuery.isError) return <div className="text-red-500">Error: {imageQuery.error.message}</div>

  return (
    <div className="space-y-4">
      <div className="text-center">
        <img 
          src={imageQuery.data?.image.image ?? ''} 
          width={256}
          height={256}
          alt={`Sample ${idx}`} 
          className="max-w-md mx-auto border rounded shadow pixelated"
        />
        <div className="mt-2 space-x-2 flex justify-center items-center">
          <Button variant="outline" onClick={prev} size="sm" disabled={idx === 0}>
            <ChevronLeft className="w-4 h-4 mr-1" /> Prev
          </Button>
          <span className="font-mono px-4 py-1 bg-muted rounded">Idx: {idx}</span>
          <Button variant="outline" onClick={next} size="sm" disabled={idx >= 9999}>
            Next <ChevronRight className="w-4 h-4 ml-1" />
          </Button>
        </div>
      </div>
      {predQuery.isSuccess && (
        <Card>
          <CardContent className="pt-6">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Predicted</TableHead>
                  <TableHead>True</TableHead>
                  <TableHead>Top Probs</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                <TableRow>
                  <TableCell className="font-semibold">{predQuery.data.prediction.pred_label}</TableCell>
                  <TableCell>{predQuery.data.prediction.true_label}</TableCell>
                  <TableCell>
                    {Object.entries(predQuery.data.prediction.probs)
                      .sort(([,a], [,b]) => b - a)
                      .map(([k, v]) => (
                        <Badge key={k} variant={predQuery.data.prediction.pred_label.toString() === k ? 'default' : 'outline'} className="mr-1">
                          {k}: {(v * 100).toFixed(1)}%
                        </Badge>
                      ))}
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
