export type ConversionResult = {
  job_id: string;
  title: string;
  composer: string;
  tempo_bpm: number;
  instrument_name: string;
  estimated_key: string;
  estimated_key_candidates: Array<{ key: string; score: number }>;
  note_confidences: Array<{
    onset_quarter: number;
    duration_quarter: number;
    type: string;
    midi: number | null;
    confidence: number;
    reattack_confidence: number | null;
    boundary_source: string;
    boundary_confidence: number;
  }>;
  note_count: number;
  artifacts: {
    musicxml_url: string;
    text_url: string;
    pitch_chart_url: string;
    result_json_url: string;
  };
};
