"use client";

import { useState } from "react";
import ConversionForm from "../components/ConversionForm";
import ConversionResults from "../components/ConversionResults";
import { API_BASE, convertAudio } from "../lib/api";
import type { ConversionResult } from "../lib/types";

export default function HomePage() {
  const [result, setResult] = useState<ConversionResult | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleConvert(formData: FormData) {
    setLoading(true);
    setResult(null);
    try {
      setResult(await convertAudio(formData));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="hero__eyebrow">Audio transcription</p>
        <h1 className="hero__title">Audio to Sheet Music</h1>
        <p className="hero__subtitle">
          Turn a clear single-line melody into printable treble-staff sheet music.
        </p>
      </header>

      <div className="layout-grid">
        <ConversionForm loading={loading} hasResult={!!result} onConvert={handleConvert} />
        <ConversionResults result={result} loading={loading} />
      </div>
      <p className="footer-note">
        Backend must be running at {API_BASE} for conversions to work.
      </p>
    </div>
  );
}
