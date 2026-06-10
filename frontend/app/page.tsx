"use client";

import {
  ChangeEvent,
  DragEvent,
  FormEvent,
  useEffect,
  useRef,
  useState
} from "react";

type Result = {
  job_id: string;
  title: string;
  composer: string;
  tempo_bpm: number;
  instrument_name: string;
  estimated_key: string;
  estimated_key_candidates: Array<{
    key: string;
    score: number;
  }>;
  note_confidences: Array<{
    onset_quarter: number;
    duration_quarter: number;
    type: string;
    midi: number | null;
    confidence: number;
    reattack_confidence: number | null;
    boundary_source: string;
    boundary_confidence: number;
    hand?: string | null;
  }>;
  chord_events: Array<{
    onset_sec: number;
    duration_sec: number;
    chord: string;
    confidence: number;
  }>;
  note_count: number;
  artifacts: {
    musicxml_url: string;
    text_url: string;
    pitch_chart_url: string;
    result_json_url: string;
    storage_path: string;
    relative_storage_path: string;
  };
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function getFriendlyHttpError(status: number): string {
  if (status === 400) return "Your upload request is invalid. Check fields and audio file.";
  if (status === 413) return "The uploaded file is too large.";
  if (status === 415) return "Unsupported file type. Try mp3, wav, or m4a.";
  if (status === 422) return "Some form values are invalid. Please review and retry.";
  if (status >= 500) return "Backend conversion crashed. Check backend terminal logs.";
  return `Request failed with status ${status}.`;
}

function statusBarClass(loading: boolean, error: string | null, hasResult: boolean): string {
  if (loading) return "status-bar status-bar--loading";
  if (error) return "status-bar status-bar--error";
  if (hasResult) return "status-bar status-bar--success";
  return "status-bar";
}

export default function HomePage() {
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusText, setStatusText] = useState("Ready — upload an audio file to begin.");
  const [rawResponse, setRawResponse] = useState("");
  const [sheetRenderError, setSheetRenderError] = useState<string | null>(null);
  const [title, setTitle] = useState("Untitled");
  const [layout, setLayout] = useState<"melody" | "grand">("melody");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const sheetContainerRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

    setLoading(true);
    setError(null);
    setResult(null);
    setStatusText("Uploading audio…");

    const formData = new FormData(event.currentTarget);
    formData.set("file", selectedFile);

    try {
      setStatusText("Analyzing pitch, rhythm, and key…");
      const response = await fetch(`${API_BASE}/jobs/convert`, {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        let backendDetail = "";
        try {
          const json = (await response.json()) as { detail?: string };
          backendDetail = json.detail ? ` ${json.detail}` : "";
        } catch {
          const text = await response.text();
          backendDetail = text ? ` ${text.slice(0, 200)}` : "";
        }
        throw new Error(`${getFriendlyHttpError(response.status)}${backendDetail}`);
      }
      const data = (await response.json()) as Result;
      setResult(data);
      setRawResponse(JSON.stringify(data, null, 2));
      setStatusText("Conversion complete.");
    } catch (err) {
      if (err instanceof TypeError) {
        setError(
          `Cannot reach backend at ${API_BASE}. Make sure the backend server is running.`
        );
      } else {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
      setStatusText("Conversion failed.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    async function renderSheetMusic() {
      if (!result?.artifacts.musicxml_url || !sheetContainerRef.current) return;
      setSheetRenderError(null);
      sheetContainerRef.current.innerHTML = "";
      try {
        const xmlText = await fetch(result.artifacts.musicxml_url).then((res) => res.text());
        const { OpenSheetMusicDisplay } = await import("opensheetmusicdisplay");
        const osmd = new OpenSheetMusicDisplay(sheetContainerRef.current, {
          autoResize: true,
          drawTitle: true
        });
        await osmd.load(xmlText);
        osmd.render();
      } catch {
        setSheetRenderError(
          "Could not render the sheet preview in your browser. Use the MusicXML download below."
        );
      }
    }

    renderSheetMusic();
  }, [result]);

  const lowConfidenceEvents = result
    ? [...result.note_confidences]
        .sort((a, b) => a.confidence - b.confidence)
        .slice(0, 6)
    : [];

  return (
    <div className="page">
      <header className="hero">
        <p className="hero__eyebrow">Audio transcription</p>
        <h1 className="hero__title">Audio to Sheet Music</h1>
        <p className="hero__subtitle">
          Upload a recording and get printable sheet music, a pitch chart, and detailed
          analysis — tuned for simple melodies or full piano arrangements.
        </p>
      </header>

      <div className="layout-grid">
        <aside className="card">
          <div className="card__header">
            <h2 className="card__title">Convert</h2>
            <p className="card__desc">Supported formats: MP3, WAV, M4A, and other audio files.</p>
          </div>
          <div className="card__body">
            <div className={statusBarClass(loading, error, !!result)}>
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

              <div className="field">
                <label className="field__label" htmlFor="layout">
                  Notation layout
                </label>
                <select
                  id="layout"
                  name="layout"
                  value={layout}
                  onChange={(event) =>
                    setLayout(event.target.value === "grand" ? "grand" : "melody")
                  }
                >
                  <option value="melody">Single staff — melody on treble</option>
                  <option value="grand">Grand staff — two hands (complex piano)</option>
                </select>
                <p className="field__hint">
                  Use single staff for tutorials and simple melodies. Use grand staff for dense
                  piano pieces.
                </p>
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

        <section className="card">
          <div className="card__header">
            <h2 className="card__title">Results</h2>
            <p className="card__desc">
              {result
                ? `${result.title}${result.composer !== "Unknown" ? ` · ${result.composer}` : ""}`
                : "Your sheet music and analysis will appear here after conversion."}
            </p>
          </div>
          <div className="card__body">
            {!result && !loading ? (
              <div className="empty-state">
                <span className="empty-state__icon" aria-hidden>
                  𝄞
                </span>
                <p className="empty-state__title">No sheet music yet</p>
                <p className="empty-state__text">
                  Upload an audio file on the left and click Generate sheet music to see your
                  score, pitch chart, and downloads.
                </p>
              </div>
            ) : null}

            {loading && !result ? (
              <div className="empty-state">
                <span className="empty-state__icon" aria-hidden>
                  ♩
                </span>
                <p className="empty-state__title">Working on it…</p>
                <p className="empty-state__text">
                  Detecting notes, estimating key, and building your score. This may take a moment
                  for longer recordings.
                </p>
              </div>
            ) : null}

            {result ? (
              <div className="result-section">
                <div className="stats">
                  <div className="stat">
                    <span className="stat__label">Key</span>
                    <span className="stat__value">{result.estimated_key}</span>
                  </div>
                  <div className="stat">
                    <span className="stat__label">Notes</span>
                    <span className="stat__value">{result.note_count}</span>
                  </div>
                  <div className="stat">
                    <span className="stat__label">Tempo</span>
                    <span className="stat__value">{result.tempo_bpm} BPM</span>
                  </div>
                  {result.chord_events?.length ? (
                    <div className="stat">
                      <span className="stat__label">Chords</span>
                      <span className="stat__value">{result.chord_events.length}</span>
                    </div>
                  ) : null}
                </div>

                <div>
                  <h3 className="section-title">Sheet music</h3>
                  <div className="preview-panel" ref={sheetContainerRef} />
                  {sheetRenderError ? (
                    <div className="alert alert--warning" role="status" style={{ marginTop: "0.75rem" }}>
                      {sheetRenderError}
                    </div>
                  ) : null}
                </div>

                {result.artifacts.pitch_chart_url ? (
                  <div>
                    <h3 className="section-title">Pitch chart</h3>
                    <div className="preview-panel">
                      <img src={result.artifacts.pitch_chart_url} alt="Detected pitch over time" />
                    </div>
                  </div>
                ) : null}

                <div>
                  <h3 className="section-title">Downloads</h3>
                  <div className="btn-group">
                    <a
                      className="btn btn--secondary"
                      href={result.artifacts.musicxml_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      MusicXML
                    </a>
                    <a
                      className="btn btn--secondary"
                      href={result.artifacts.text_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Text score
                    </a>
                    <a
                      className="btn btn--secondary"
                      href={result.artifacts.pitch_chart_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Pitch chart
                    </a>
                    <a
                      className="btn btn--secondary"
                      href={result.artifacts.result_json_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      JSON data
                    </a>
                  </div>
                </div>

                {result.estimated_key_candidates?.length ? (
                  <details className="details">
                    <summary>Key analysis</summary>
                    <div className="details__body">
                      <ul className="key-list">
                        {result.estimated_key_candidates.map((candidate) => (
                          <li className="key-list__item" key={candidate.key}>
                            <span>{candidate.key}</span>
                            <div className="key-list__bar-wrap">
                              <div
                                className="key-list__bar"
                                style={{ width: `${Math.min(100, candidate.score * 100)}%` }}
                              />
                            </div>
                            <span>{(candidate.score * 100).toFixed(0)}%</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </details>
                ) : null}

                {result.chord_events?.length ? (
                  <details className="details">
                    <summary>Detected chords ({result.chord_events.length})</summary>
                    <div className="details__body">
                      <ul>
                        {result.chord_events.slice(0, 12).map((chord, idx) => (
                          <li key={`${chord.onset_sec}-${idx}`}>
                            {chord.chord} at {chord.onset_sec.toFixed(1)}s for{" "}
                            {chord.duration_sec.toFixed(1)}s ·{" "}
                            {(chord.confidence * 100).toFixed(0)}% confidence
                          </li>
                        ))}
                      </ul>
                    </div>
                  </details>
                ) : null}

                {lowConfidenceEvents.length ? (
                  <details className="details">
                    <summary>Lowest-confidence notes</summary>
                    <div className="details__body">
                      <p>These events may need manual review in the final score.</p>
                      <ul>
                        {lowConfidenceEvents.map((event, idx) => (
                          <li key={`${event.onset_quarter}-${idx}`}>
                            {event.type === "note" && event.midi !== null
                              ? `MIDI ${event.midi}`
                              : "Rest"}{" "}
                            · beat {event.onset_quarter.toFixed(2)} ·{" "}
                            {(event.confidence * 100).toFixed(0)}% confidence
                            {event.hand ? ` · ${event.hand} hand` : ""}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </details>
                ) : null}

                <details className="details">
                  <summary>Technical details</summary>
                  <div className="details__body">
                    <p>
                      <strong>Job ID:</strong> {result.job_id}
                    </p>
                    <p>
                      <strong>Storage:</strong>{" "}
                      <code>{result.artifacts.relative_storage_path}</code>
                    </p>
                    <pre className="code-block">{rawResponse}</pre>
                  </div>
                </details>
              </div>
            ) : null}
          </div>
        </section>
      </div>

      <p className="footer-note">
        Backend must be running at {API_BASE} for conversions to work.
      </p>
    </div>
  );
}
