import type { ConversionResult } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function getFriendlyHttpError(status: number): string {
  if (status === 400) return "Your upload request is invalid. Check fields and audio file.";
  if (status === 413) return "The uploaded file is too large.";
  if (status === 415) return "Unsupported file type. Try mp3, wav, or m4a.";
  if (status === 422) return "Some form values are invalid. Please review and retry.";
  if (status >= 500) return "Backend conversion crashed. Check backend terminal logs.";
  return `Request failed with status ${status}.`;
}

export async function convertAudio(formData: FormData): Promise<ConversionResult> {
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

  return (await response.json()) as ConversionResult;
}
