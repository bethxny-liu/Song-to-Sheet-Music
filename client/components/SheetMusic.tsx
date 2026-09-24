"use client";

import { useEffect, useRef, useState } from "react";
import { exportSheetMusicPdf, type OsmdInstance } from "../lib/exportPdf";

export default function SheetMusic({ musicxmlUrl, title }: { musicxmlUrl: string; title: string }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const osmdRef = useRef<OsmdInstance | null>(null);
  const [ready, setReady] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const controller = new AbortController();
    let renderer: { clear: () => void; setOptions: (options: { autoResize: boolean }) => void } | null = null;
    setReady(false);
    setError(null);

    async function render() {
      try {
        const response = await fetch(musicxmlUrl, { signal: controller.signal });
        if (!response.ok) throw new Error(`MusicXML request failed: ${response.status}`);
        const xml = await response.text();
        const { OpenSheetMusicDisplay, BackendType } = await import("opensheetmusicdisplay");
        if (controller.signal.aborted) return;
        const osmd = new OpenSheetMusicDisplay(container, {
          autoResize: true,
          drawTitle: true,
          backendType: BackendType.SVG
        });
        renderer = osmd;
        await osmd.load(xml);
        if (controller.signal.aborted) return;
        osmd.render();
        osmdRef.current = osmd;
        setReady(true);
      } catch {
        if (!controller.signal.aborted) {
          setError("Could not render the sheet preview in your browser. Use the MusicXML download below.");
        }
      }
    }

    void render();
    return () => {
      controller.abort();
      renderer?.setOptions({ autoResize: false });
      renderer?.clear();
      osmdRef.current = null;
      container.replaceChildren();
    };
  }, [musicxmlUrl]);

  async function downloadPdf() {
    if (!osmdRef.current) return;
    setExporting(true);
    try {
      const safeTitle = title.replace(/[^\w\s-]/g, "").trim() || "sheet-music";
      await exportSheetMusicPdf(osmdRef.current, `${safeTitle}.pdf`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not export PDF. Try the MusicXML download instead.");
    } finally {
      setExporting(false);
    }
  }

  return (
    <div>
      <h3 className="section-title">Sheet music</h3>
      <div className="preview-panel" ref={containerRef} />
      {error ? <div className="alert alert--warning" role="status" style={{ marginTop: "0.75rem" }}>{error}</div> : null}
      <button className="btn btn--secondary" type="button" onClick={downloadPdf}
        disabled={!ready || exporting || !!error} style={{ marginTop: "0.75rem" }}>
        {exporting ? "Exporting PDF…" : "Download PDF"}
      </button>
    </div>
  );
}
