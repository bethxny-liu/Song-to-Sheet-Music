import { jsPDF } from "jspdf";

type OsmdInstance = {
  backendType: number;
  drawer: { Backends: Array<{ getSvgElement: () => SVGElement }> };
  rules: { PageFormat?: { IsUndefined?: boolean; width: number; height: number } };
  sheet?: { FullNameString?: string };
};

const SVG_BACKEND = 0;

function svgElementToDataUrl(svgElement: SVGElement, scale = 2, jpegQuality = 0.9): Promise<string> {
  return new Promise((resolve, reject) => {
    const clone = svgElement.cloneNode(true) as SVGElement;
    if (!clone.getAttribute("xmlns")) {
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    }
    if (!clone.getAttribute("xmlns:xlink")) {
      clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
    }

    const width = svgElement.clientWidth || svgElement.getBoundingClientRect().width;
    const height = svgElement.clientHeight || svgElement.getBoundingClientRect().height;
    clone.setAttribute("width", String(width));
    clone.setAttribute("height", String(height));

    const svgData = new XMLSerializer().serializeToString(clone);
    const url = URL.createObjectURL(new Blob([svgData], { type: "image/svg+xml;charset=utf-8" }));

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width * scale;
      canvas.height = height * scale;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        reject(new Error("Could not create canvas context."));
        return;
      }
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0, width, height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", jpegQuality));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not rasterize sheet music for PDF export."));
    };
    img.src = url;
  });
}

export async function exportSheetMusicPdf(osmd: OsmdInstance, pdfName: string): Promise<void> {
  if (osmd.backendType !== SVG_BACKEND) {
    throw new Error("PDF export requires SVG rendering.");
  }

  const backends = osmd.drawer.Backends;
  if (!backends.length) {
    throw new Error("No sheet music pages to export.");
  }

  const firstSvg = backends[0].getSvgElement();
  let pageWidth = 210;
  let pageHeight = 297;
  const pageFormat = osmd.rules.PageFormat;
  if (pageFormat && !pageFormat.IsUndefined) {
    pageWidth = pageFormat.width;
    pageHeight = pageFormat.height;
  } else {
    pageHeight = pageWidth * (firstSvg.clientHeight / firstSvg.clientWidth);
  }

  const orientation = pageHeight > pageWidth ? "p" : "l";
  const pdf = new jsPDF({ orientation, unit: "mm", format: [pageWidth, pageHeight] });

  for (let idx = 0; idx < backends.length; idx++) {
    if (idx > 0) {
      pdf.addPage();
    }
    const imageDataUrl = await svgElementToDataUrl(backends[idx].getSvgElement());
    pdf.addImage(imageDataUrl, "JPEG", 0, 0, pageWidth, pageHeight);
  }

  pdf.save(pdfName.endsWith(".pdf") ? pdfName : `${pdfName}.pdf`);
}
