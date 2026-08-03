/**
 * OCR-routed extensions (mcp_server/ocr.py — Apple Vision + PyMuPDF). Must
 * stay in sync with the backend's OCR_MIME_BY_EXTENSION
 * (mcp_server/ocr.py) and session_files.py's ALLOWED_EXTENSIONS.
 */
export const OCR_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.heic', '.pdf']);

export function extOf(filename: string): string {
  return '.' + (filename.split('.').pop()?.toLowerCase() ?? '');
}

export function isOcrExtension(filename: string): boolean {
  return OCR_EXTENSIONS.has(extOf(filename));
}
