"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useRef, useState } from "react";

function statusBarClass(loading: boolean, error: string | null, hasResult: boolean): string {
  if (loading) return "status-bar status-bar--loading";
  if (error) return "status-bar status-bar--error";
  if (hasResult) return "status-bar status-bar--success";
  return "status-bar";
}

export default function ConversionForm({ loading, hasResult, onConvert }: {
  loading: boolean;
  hasResult: boolean;
  onConvert: (formData: FormData) => Promise<void>;
}) {
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("Ready — upload an audio file to begin.");
  const [title, setTitle] = useState("Untitled");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!loading) return;

    const startedAt = Date.now();
    const updateStatus = () => {
      const elapsedSeconds = (Date.now() - startedAt) / 1000;
      if (elapsedSeconds >= 20) {
        setStatusText("Still working — longer recordings can take a few minutes.");
      } else if (elapsedSeconds >= 8) {
        setStatusText("Analyzing the melody and preparing the score…");
      } else {
        setStatusText("Analyzing pitch, rhythm, and key…");
      }
    };

    updateStatus();
    const timer = window.setInterval(updateStatus, 1000);
    return () => window.clearInterval(timer);
  }, [loading]);

  function applySelectedFile(file: File | undefined) {
    if (!file) return;
    setSelectedFile(file);
    const baseName = file.name.replace(/\.[^/.]+$/, "").trim();
    if (baseName) setTitle(baseName);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    applySelectedFile(event.target.files?.[0]);
  }

  function handleDragOver(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragLeave(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragActive(false);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith("audio/")) {
      applySelectedFile(file);
      if (fileInputRef.current) {
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInputRef.current.files = dt.files;
      }
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please choose an audio file first.");
      return;
    }

    setError(null);
    setStatusText("Uploading audio…");

    const formData = new FormData(event.currentTarget);
    formData.set("file", selectedFile);

    try {
      setStatusText("Analyzing pitch, rhythm, and key…");
      await onConvert(formData);
      setStatusText("Conversion complete.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Conversion failed. Please try again.");
      setStatusText("Conversion failed.");
    }
  }

  return (
    <aside className="card">
      <div className="card__header">
        <h2 className="card__title">Convert</h2>
        <p className="card__desc">MP3, WAV, M4A, and other decodable audio. Up to 25 MB and 4 minutes.</p>
      </div>
      <div className="card__body">
        <div className={statusBarClass(loading, error, hasResult)}>
          <span className="status-dot" aria-hidden />
          <span>{statusText}</span>
        </div>

        <form className="form" onSubmit={handleSubmit}>
          <div className="field">
            <span className="field__label">Audio file</span>
            <label
              className={`dropzone${dragActive ? " dropzone--active" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <span className="dropzone__icon" aria-hidden>
                ♪
              </span>
              <p className="dropzone__title">
                {selectedFile ? "Replace audio file" : "Drop audio here or click to browse"}
              </p>
              <p className="dropzone__subtitle">One file at a time</p>
              {selectedFile ? (
                <p className="dropzone__file">{selectedFile.name}</p>
              ) : null}
              <input
                ref={fileInputRef}
                name="file"
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                required
              />
            </label>
          </div>

          <div className="field">
            <label className="field__label" htmlFor="title">
              Title
            </label>
            <input
              id="title"
              name="title"
              type="text"
              placeholder="Song title"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
            />
          </div>

          <div className="field">
            <label className="field__label" htmlFor="composer">
              Composer
            </label>
            <input
              id="composer"
              name="composer"
              type="text"
              placeholder="Unknown"
              defaultValue="Unknown"
            />
          </div>

          <div className="field-row">
            <div className="field">
              <label className="field__label" htmlFor="tempo_bpm">
                Tempo (BPM)
              </label>
              <input
                id="tempo_bpm"
                name="tempo_bpm"
                type="number"
                min={40}
                max={240}
                defaultValue={90}
              />
            </div>
            <div className="field">
              <label className="field__label" htmlFor="instrument_name">
                Instrument
              </label>
              <input
                id="instrument_name"
                name="instrument_name"
                type="text"
                defaultValue="piano"
              />
            </div>
          </div>

          <button className="btn" type="submit" disabled={loading || !selectedFile}>
            {loading ? (
              <>
                <span className="spinner" aria-hidden />
                Converting…
              </>
            ) : (
              "Generate sheet music"
            )}
          </button>
        </form>

        {error ? (
          <div className="alert alert--error" role="alert" style={{ marginTop: "1rem" }}>
            {error}
          </div>
        ) : null}
      </div>
    </aside>
  );
}
