import type { ConversionResult } from "../lib/types";
import SheetMusic from "./SheetMusic";

export default function ConversionResults({ result, loading }: { result: ConversionResult | null; loading: boolean }) {
  const lowConfidenceEvents = result
    ? [...result.note_confidences]
        .sort((a, b) => a.confidence - b.confidence)
        .slice(0, 6)
    : [];

  return (
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
              Detecting notes, estimating key, and building your score. Longer recordings can
              take a few minutes.
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
            </div>

            <SheetMusic key={result.job_id} musicxmlUrl={result.artifacts.musicxml_url} title={result.title} />

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
                  <a href={result.artifacts.result_json_url} target="_blank" rel="noreferrer">
                    View full JSON response
                  </a>
                </p>
              </div>
            </details>
          </div>
        ) : null}
      </div>
    </section>
  );
}
