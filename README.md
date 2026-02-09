# Liveness Web

## Project Overview
`liveness-web` is a web-based liveness verification UI and engine integration. It uses the `cynopsis-vecface-frond` liveness engine (SCRFD + PFLD ONNX) to guide users through four actions (nod, shake, blink, open mouth), captures evidence GIFs, and outputs structured results to the browser console for downstream systems to collect.

Key capabilities:
- Automatic camera startup and model warmup on page load.
- Four-step liveness flow with timed action verification.
- Evidence GIF generation per action.
- Face images captured as full-frame and cropped face.
- Final result is emitted as JSON to the console.
- HTTPS enabled for LAN access (via Vite basic SSL).

## How It Works (High Level)
1. **Camera start**: The page starts the camera automatically.
2. **Model warmup**: ONNX models load in browser.
3. **Action verification**: The engine evaluates 4 actions in order.
4. **Evidence generation**: Each action produces a GIF (stored in memory).
5. **Result output**: On completion (PASS/FAIL), GIFs and face images are downloaded, and a JSON payload is printed to the console.

## Running Locally
```bash
cd /Users/wangzai/workspace/showcase_liveness_web/liveness-web
npm install
npm run dev
```

### LAN Access (HTTPS)
Vite is configured with HTTPS and `host: true`. Access via:
```
https://<your-lan-ip>:5173/
```
You will need to accept the self-signed certificate in the browser.

## Output Format (Console JSON)
The result is printed to the console with a stable prefix:
```
[LIVENESS_RESULT] {JSON}
```

### JSON Schema
```json
{
  "event": "LIVENESS_RESULT",
  "status": "PASS | FAIL",
  "sessionState": "IDLE | READY | LOADING | IN_PROGRESS | COMPLETED | FAILED",
  "scores": {
    "qualityScore": number | null,
    "brightnessScore": number | null,
    "brightnessStatus": string | null,
    "frontalFaceScore": number | null,
    "clarityScore": number | null,
    "uniformLightingScore": number | null,
    "backgroundUniformityScore": number | null,
    "pixelResolutionScore": number | null
  },
  "images": {
    "faceFull": "data:image/png;base64,..." | null,
    "faceCrop": "data:image/png;base64,..." | null,
    "gifs": {
      "nod": "data:image/gif;base64,..." | null,
      "shake": "data:image/gif;base64,..." | null,
      "blink": "data:image/gif;base64,..." | null,
      "open": "data:image/gif;base64,..." | null
    }
  }
}
```

## Evidence File Download
On completion, files are downloaded automatically:
- `nod-<timestamp>.gif`
- `shake-<timestamp>.gif`
- `blink-<timestamp>.gif`
- `open-<timestamp>.gif`
- `face-<timestamp>.png`

## Source Layout
```
liveness-web/
  src/
    components/FaceLiveness.tsx
    services/livenessEngine.ts
    models/*.onnx
    assets/onnxruntime/*
```

## Notes
- GIFs are generated in-memory via `gifshot` and emitted as data URLs.
- If the host app needs structured output, parse console logs by prefix `[LIVENESS_RESULT]`.
