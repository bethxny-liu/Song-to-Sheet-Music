import type { ConversionResult } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function getFriendlyHttpError(status: number): string {
  if (status === 400) return "Your upload request is invalid. Check fields and audio file.";
  if (status === 413) return "The uploaded file is too large.";
  if (status === 415) return "Unsupported file type. Try mp3, wav, or m4a.";
  if (status === 422) return "Some form values are invalid. Please review and retry.";
  if (status >= 500) return "Conversion failed on the server. Please try again.";
  return `Request failed with status ${status}.`;
}

export async function convertAudio(formData: FormData): Promise<ConversionResult> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/jobs/convert`, {
      method: "POST",
      body: formData
    });
  } catch {
    throw new Error(`Cannot reach backend at ${API_BASE}. Make sure the backend server is running.`);
  }

  if (!response.ok) {
    const text = await response.text();
    let message = getFriendlyHttpError(response.status);
    try {
      const json = JSON.parse(text) as { detail?: unknown };
      if (typeof json.detail === "string") {
        message = json.detail;
      } else if (Array.isArray(json.detail)) {
        const errors = json.detail
          .filter((error) => typeof error?.msg === "string")
          .map((error) => {
            const field = Array.isArray(error.loc) ? error.loc.filter((part: unknown) => part !== "body").join(".") : "";
            return field ? `${field}: ${error.msg}` : error.msg;
          });
        if (errors.length) message = errors.join(" ");
      }
    } catch {
      // Plain-text or HTML error pages use the friendly status message.
    }
    throw new Error(message);
  }

  try {
    return (await response.json()) as ConversionResult;
  } catch {
    throw new Error("The server returned an unreadable conversion result. Please try again.");
  }
}
