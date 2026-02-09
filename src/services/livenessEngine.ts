import * as ort from 'onnxruntime-web';
import ortWasmSimdThreadedJsepMjs from '../assets/onnxruntime/ort-wasm-simd-threaded.jsep.mjs?url';
import ortWasmSimdThreadedJsepWasm from '../assets/onnxruntime/ort-wasm-simd-threaded.jsep.wasm?url';
import ortWasmSimdThreadedMjs from '../assets/onnxruntime/ort-wasm-simd-threaded.mjs?url';
import ortWasmSimdThreadedWasm from '../assets/onnxruntime/ort-wasm-simd-threaded.wasm?url';
import ortWasmSimdThreadedAsyncifyMjs from '../assets/onnxruntime/ort-wasm-simd-threaded.asyncify.mjs?url';
import ortWasmSimdThreadedAsyncifyWasm from '../assets/onnxruntime/ort-wasm-simd-threaded.asyncify.wasm?url';
import scrfdModelUrl from '../models/scrfd_500m_bnkps_shape640x640.onnx?url';
import pfpldModelUrl from '../models/pfpld.onnx?url';

const SCRFD_RELATIVE_PATH = 'models/scrfd_500m_bnkps_shape640x640.onnx';
const PFPLD_RELATIVE_PATH = 'models/pfpld.onnx';
const DEFAULT_MODEL_URL = scrfdModelUrl;
const DEFAULT_LANDMARK_MODEL_URL = pfpldModelUrl;

type ExecutionProvider = 'webgpu' | 'webgl' | 'wasm';

const EXECUTION_PROVIDERS: ExecutionProvider[] = ['webgpu', 'webgl', 'wasm'];

export const LIVENESS_CONFIG = {
  nodPitch: 0.05,
  shakeYaw: 0.06,
  blinkThreshold: 0.018,
  mouthThreshold: 0.5,
};

const INITIAL_BLINK_AVG = 10;
const BLINK_THRESHOLD_LIMITS = { low: 5, high: 20 };
const BLINK_AVG_WEIGHTS = { current: 0.1, previous: 0.9 };
const POSE_SCORE_SIGMA = 20;
const POSE_SCORE_WEIGHTS = { yaw: 1, pitch: 1, roll: 0.8 };

type ModelSource = string | ArrayBuffer | Blob | string[];

const ORT_WASM_PATHS = {
  mjs: ortWasmSimdThreadedJsepMjs,
  wasm: ortWasmSimdThreadedJsepWasm,
  fallbackMjs: ortWasmSimdThreadedMjs,
  fallbackWasm: ortWasmSimdThreadedWasm,
  asyncifyMjs: ortWasmSimdThreadedAsyncifyMjs,
  asyncifyWasm: ortWasmSimdThreadedAsyncifyWasm,
};

function normalizeRelativePath(path: string) {
  return path.replace(/^(?:\.\.?\/)+/, '').replace(/^\/+/, '');
}

function buildModelSourceCandidates(primary: string, relativePath: string) {
  const normalized = normalizeRelativePath(relativePath);
  const candidates = [
    primary,
    relativePath,
    `./${normalized}`,
    `/${normalized}`,
  ];
  return Array.from(new Set(candidates.filter((item) => typeof item === 'string' && item.length)));
}

async function fetchModelFromUrl(url: string) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`);
  }
  const buffer = await response.arrayBuffer();
  const headerBytes = new Uint8Array(buffer.slice(0, 32));
  const headerText = new TextDecoder('utf-8', { fatal: false }).decode(headerBytes);
  const contentType = response.headers.get('content-type') || '';
  const lowerHeader = headerText.toLowerCase();
  if (
    contentType.includes('text/html') ||
    lowerHeader.startsWith('<!doctype') ||
    lowerHeader.startsWith('<html') ||
    lowerHeader.startsWith('<!do') ||
    lowerHeader.startsWith('<!ht')
  ) {
    throw new Error(`Model URL returned HTML instead of ONNX: ${url}`);
  }
  return buffer;
}

export type LivenessStageKey = 'nod' | 'shake' | 'blink' | 'mouth';

interface Detection {
  bbox: [number, number, number, number];
  score: number;
  kps?: Array<{ x: number; y: number }>;
}

interface DetectionWithLandmarks extends Detection {
  landmarks: Array<{ x: number; y: number }>;
  pose: Pose | null;
}

export interface Pose {
  yaw: number;
  pitch: number;
  roll: number;
}

export interface LivenessMetrics {
  nodRatio: number | null;
  nodSpread: number | null;
  shakeRatio: number | null;
  shakeSpread: number | null;
  blink: ReturnType<typeof computeBlinkRatios>;
  mouthRatio: number | null;
  pose: Pose | null;
  poseDegrees: Pose | null;
  frontalScore: number | null;
}

export interface FaceDetectionResult {
  bbox: [number, number, number, number];
  score: number;
  kps?: Array<{ x: number; y: number }>;
}

interface BlinkState {
  closedDetected: boolean;
  openDetected: boolean;
  eyeAvg: number | null;
}

interface MouthState {
  openDetected: boolean;
  closedDetected: boolean;
}

interface RangeState {
  min: number | null;
  max: number | null;
}

interface LivenessComputationState {
  active: boolean;
  stageIndex: number;
  progress: number;
  completed: boolean;
  lastMetrics: LivenessMetrics | null;
  nodRange: RangeState;
  shakeRange: RangeState;
  blink: BlinkState;
  mouth: MouthState;
}

function createLivenessComputationState(): LivenessComputationState {
  return {
    active: false,
    stageIndex: 0,
    progress: 0,
    completed: false,
    lastMetrics: null,
    nodRange: { min: null, max: null },
    shakeRange: { min: null, max: null },
    blink: {
      closedDetected: false,
      openDetected: false,
      eyeAvg: INITIAL_BLINK_AVG,
    },
    mouth: {
      openDetected: false,
      closedDetected: false,
    },
  };
}

export interface LivenessStageDefinition {
  key: LivenessStageKey;
  label: string;
  prompt: string;
  check: (metrics: LivenessMetrics, state: LivenessComputationState) => boolean;
}

export const LIVENESS_STAGES: LivenessStageDefinition[] = [
  {
    key: 'nod',
    label: '点头',
    prompt: '请点头',
    check: (metrics) =>
      typeof metrics.nodSpread === 'number' && metrics.nodSpread >= LIVENESS_CONFIG.nodPitch,
  },
  {
    key: 'shake',
    label: '摇头',
    prompt: '请左右摇头',
    check: (metrics) =>
      typeof metrics.shakeSpread === 'number' && metrics.shakeSpread >= LIVENESS_CONFIG.shakeYaw,
  },
  {
    key: 'blink',
    label: '眨眼',
    prompt: '请眨眼',
    check: (_, state) => state.blink.closedDetected && state.blink.openDetected,
  },
  {
    key: 'mouth',
    label: '张嘴',
    prompt: '请张大嘴巴',
    check: (_, state) => state.mouth.closedDetected && state.mouth.openDetected,
  },
];

type InputElement = HTMLVideoElement | HTMLCanvasElement | HTMLImageElement;

function getExecutionProviders(): ExecutionProvider[] {
  return EXECUTION_PROVIDERS.slice();
}

function isSimdUnavailableError(err: unknown) {
  const msg = err instanceof Error ? err.message : '';
  return /simd/i.test(msg) || /no available backend/i.test(msg);
}

function isSafari() {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
  if (!ua) return false;
  return /safari/i.test(ua) && !/chrome|chromium|edg/i.test(ua);
}

function configureOrtEnv() {
  if (!ort.env || !ort.env.wasm) return;
  const wasmPaths = ort.env.wasm.wasmPaths;
  if (!wasmPaths || typeof wasmPaths === 'string') {
    ort.env.wasm.wasmPaths = {
      mjs: ORT_WASM_PATHS.mjs,
      wasm: ORT_WASM_PATHS.wasm,
    };
    return;
  }
  if (typeof wasmPaths === 'object') {
    const wasmPathMap = wasmPaths as Record<string, string>;
    const missingKey = !wasmPathMap.mjs || !wasmPathMap.wasm;
    if (missingKey) {
      ort.env.wasm.wasmPaths = {
        mjs: ORT_WASM_PATHS.mjs,
        wasm: ORT_WASM_PATHS.wasm,
        ...wasmPathMap,
      };
    }
  }
}

function isHuaweiWebgpuRestricted() {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : '';
  return /huawei|kirin/i.test(ua);
}

function forceScalarWasmBackend() {
  if (!ort.env || !ort.env.wasm) return;
  ort.env.wasm.simd = false;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
}

function applyDeviceBackendOverrides() {
  if (isHuaweiWebgpuRestricted()) {
    EXECUTION_PROVIDERS.splice(0, EXECUTION_PROVIDERS.length, 'webgl', 'wasm');
  }
}

async function createOrtSession(buffer: ArrayBuffer) {
  try {
    return await ort.InferenceSession.create(buffer, {
      executionProviders: getExecutionProviders(),
    });
  } catch (err) {
    if (isSimdUnavailableError(err)) {
      forceScalarWasmBackend();
      return ort.InferenceSession.create(buffer, {
        executionProviders: getExecutionProviders(),
      });
    }
    throw err;
  }
}

class SCRFDDetector {
  private session: ort.InferenceSession | null = null;
  private inputName: string | null = null;
  private outputNames: string[] = [];
  private inputSize: [number, number] = [640, 640];
  private centerCache: Map<string, Float32Array> = new Map();
  private useKps = false;
  private _numAnchors = 1;
  private _featStrideFpn = [8, 16, 32];
  private fmc = 3;
  private batched = false;
  private preprocessCanvas: HTMLCanvasElement;
  private preCtx: CanvasRenderingContext2D;
  private nmsThresh = 0.4;
  private kpsPerAnchor = 0;
  private modelBuffer: ArrayBuffer | null = null;

  constructor() {
    this.preprocessCanvas = document.createElement('canvas');
    const ctx = this.preprocessCanvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      throw new Error('无法创建检测器预处理上下文');
    }
    this.preCtx = ctx;
  }

  isReady() {
    return Boolean(this.session);
  }

  async loadModel(source: ModelSource) {
    const buffer = await this.asArrayBuffer(source);
    this.modelBuffer = buffer;
    this.session = await createOrtSession(buffer);
    this.inputName = this.session.inputNames[0];
    this.outputNames = this.session.outputNames;
    const inputMeta = this.session.inputMetadata[this.inputName];
    const shape = inputMeta?.dimensions ?? [1, 3, 640, 640];
    if (typeof shape[2] === 'number' && typeof shape[3] === 'number') {
      this.inputSize = [shape[3], shape[2]];
    }
    this.initFromOutputs();
  }

  private async asArrayBuffer(source: ModelSource) {
    const candidates = Array.isArray(source) ? source : [source];
    const errors: string[] = [];
    for (const entry of candidates) {
      try {
        if (entry instanceof ArrayBuffer) {
          return entry;
        }
        if (entry instanceof Blob) {
          return entry.arrayBuffer();
        }
        if (typeof entry === 'string') {
          return await fetchModelFromUrl(entry);
        }
      } catch (err) {
        errors.push(err instanceof Error ? err.message : String(err));
      }
    }
    throw new Error(errors.length ? errors.join('; ') : '不支持的模型来源');
  }

  private initFromOutputs() {
    if (!this.session) return;
    const outCount = this.outputNames.length;
    const meta = this.session.outputMetadata[this.outputNames[0]];
    const dims = meta?.dimensions ?? [];
    this.batched = dims.length === 3;
    if (outCount === 6) {
      this.fmc = 3;
      this._featStrideFpn = [8, 16, 32];
      this._numAnchors = 2;
    } else if (outCount === 9) {
      this.fmc = 3;
      this._featStrideFpn = [8, 16, 32];
      this._numAnchors = 2;
      this.useKps = true;
    } else if (outCount === 10) {
      this.fmc = 5;
      this._featStrideFpn = [8, 16, 32, 64, 128];
      this._numAnchors = 1;
    } else if (outCount === 15) {
      this.fmc = 5;
      this._featStrideFpn = [8, 16, 32, 64, 128];
      this._numAnchors = 1;
      this.useKps = true;
    } else {
      console.warn('未知的检测器输出数量，使用默认配置');
    }
  }

  async detectFromElement(element: InputElement, threshold = 0.5) {
    if (!this.session || !this.inputName) {
      throw new Error('检测模型尚未加载');
    }
    const prep = this.prepareInput(element);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.inputName] = prep.tensor;
    const result = await this.session.run(feeds);
    const decoded = this.postprocess(result, threshold);
    return this.buildDetections(decoded, prep.detScale);
  }

  private prepareInput(element: InputElement) {
    const width = (element as HTMLVideoElement).videoWidth || element.width;
    const height = (element as HTMLVideoElement).videoHeight || element.height;
    const [targetW, targetH] = this.inputSize;
    const imRatio = height / width;
    const modelRatio = targetH / targetW;

    let newWidth: number;
    let newHeight: number;
    if (imRatio > modelRatio) {
      newHeight = targetH;
      newWidth = Math.round(newHeight / imRatio);
    } else {
      newWidth = targetW;
      newHeight = Math.round(newWidth * imRatio);
    }

    this.preprocessCanvas.width = targetW;
    this.preprocessCanvas.height = targetH;
    this.preCtx.fillStyle = 'black';
    this.preCtx.fillRect(0, 0, targetW, targetH);
    this.preCtx.drawImage(element, 0, 0, width, height, 0, 0, newWidth, newHeight);

    const imageData = this.preCtx.getImageData(0, 0, targetW, targetH).data;
    const area = targetW * targetH;
    const floatData = new Float32Array(3 * area);
    for (let i = 0; i < area; i += 1) {
      const base = i * 4;
      const r = imageData[base];
      const g = imageData[base + 1];
      const b = imageData[base + 2];
      floatData[i] = (r - 127.5) / 128;
      floatData[i + area] = (g - 127.5) / 128;
      floatData[i + area * 2] = (b - 127.5) / 128;
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, targetH, targetW]);
    const detScale = newHeight / height;
    return { tensor, detScale };
  }

  private postprocess(result: ort.InferenceSession.OnnxValueMapType, threshold: number) {
    const netOuts = this.outputNames.map((name) => result[name]);
    const scoresList: Float32Array[] = [];
    const bboxList: Float32Array[] = [];
    const kpsList: Float32Array[] = [];
    const inputHeight = this.inputSize[1];
    const inputWidth = this.inputSize[0];

    for (let idx = 0; idx < this.fmc; idx += 1) {
      const stride = this._featStrideFpn[idx];
      const height = Math.floor(inputHeight / stride);
      const width = Math.floor(inputWidth / stride);
      const anchorCenters = this.getAnchorCenters(height, width, stride);
      const K = height * width * this._numAnchors;

      const scoresTensor = netOuts[idx];
      const bboxTensor = netOuts[idx + this.fmc];
      const scores = this.extractScores(this.toFloat32(scoresTensor), K);
      const bboxPreds = this.toFloat32(bboxTensor, K * 4);
      for (let i = 0; i < bboxPreds.length; i += 1) {
        bboxPreds[i] *= stride;
      }

      const decodedBboxes = this.distance2bbox(anchorCenters, bboxPreds);
      const pos: number[] = [];
      for (let i = 0; i < K; i += 1) {
        if (scores[i] >= threshold) {
          pos.push(i);
        }
      }
      if (!pos.length) continue;

      const posScores = new Float32Array(pos.length);
      const posBoxes = new Float32Array(pos.length * 4);
      pos.forEach((pi, order) => {
        posScores[order] = scores[pi];
        const boxBase = pi * 4;
        posBoxes.set(decodedBboxes.subarray(boxBase, boxBase + 4), order * 4);
      });
      scoresList.push(posScores);
      bboxList.push(posBoxes);

      if (this.useKps) {
        const kpsTensor = netOuts[idx + this.fmc * 2];
        const kpsPreds = this.toFloat32(kpsTensor);
        for (let i = 0; i < kpsPreds.length; i += 1) {
          kpsPreds[i] *= stride;
        }
        const decodedKps = this.distance2kps(anchorCenters, kpsPreds);
        this.kpsPerAnchor = decodedKps.numKeypoints;
        const coordsPerAnchor = this.kpsPerAnchor * 2;
        const posKps = new Float32Array(pos.length * coordsPerAnchor);
        pos.forEach((pi, order) => {
          const start = pi * coordsPerAnchor;
          posKps.set(decodedKps.data.subarray(start, start + coordsPerAnchor), order * coordsPerAnchor);
        });
        kpsList.push(posKps);
      }
    }

    return {
      scores: this.concatFloat32(scoresList),
      boxes: this.concatFloat32(bboxList),
      kps: this.useKps ? this.concatFloat32(kpsList) : null,
    };
  }

  private buildDetections(decoded: { scores: Float32Array | null; boxes: Float32Array | null; kps: Float32Array | null }, detScale: number): Detection[] {
    const results: Detection[] = [];
    const { scores, boxes, kps } = decoded;
    if (!scores || !scores.length || !boxes) {
      return results;
    }
    const num = scores.length;
    const coordsPerAnchor = this.kpsPerAnchor * 2;
    for (let i = 0; i < num; i += 1) {
      const base = i * 4;
      const det: Detection = {
        bbox: [
          boxes[base] / detScale,
          boxes[base + 1] / detScale,
          boxes[base + 2] / detScale,
          boxes[base + 3] / detScale,
        ],
        score: scores[i],
      };
      if (this.useKps && kps && coordsPerAnchor) {
        const kpsStart = i * coordsPerAnchor;
        det.kps = [];
        for (let j = 0; j < coordsPerAnchor; j += 2) {
          det.kps.push({
            x: kps[kpsStart + j] / detScale,
            y: kps[kpsStart + j + 1] / detScale,
          });
        }
      }
      results.push(det);
    }
    return this.nms(results);
  }

  private getAnchorCenters(height: number, width: number, stride: number) {
    const key = `${height}-${width}-${stride}`;
    if (this.centerCache.has(key)) {
      return this.centerCache.get(key)!;
    }
    const anchorCount = height * width * this._numAnchors;
    const centers = new Float32Array(anchorCount * 2);
    let ptr = 0;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        for (let a = 0; a < this._numAnchors; a += 1) {
          centers[ptr] = x * stride;
          centers[ptr + 1] = y * stride;
          ptr += 2;
        }
      }
    }
    if (this.centerCache.size < 100) {
      this.centerCache.set(key, centers);
    }
    return centers;
  }

  private distance2bbox(points: Float32Array, distance: Float32Array) {
    const anchorCount = points.length / 2;
    const output = new Float32Array(anchorCount * 4);
    for (let i = 0; i < anchorCount; i += 1) {
      const px = points[i * 2];
      const py = points[i * 2 + 1];
      const base = i * 4;
      const left = distance[base];
      const top = distance[base + 1];
      const right = distance[base + 2];
      const bottom = distance[base + 3];
      output[base] = px - left;
      output[base + 1] = py - top;
      output[base + 2] = px + right;
      output[base + 3] = py + bottom;
    }
    return output;
  }

  private distance2kps(points: Float32Array, distance: Float32Array) {
    const anchorCount = points.length / 2;
    const coordsPerAnchor = distance.length / anchorCount;
    const numKeypoints = Math.round(coordsPerAnchor / 2);
    const output = new Float32Array(distance.length);
    for (let i = 0; i < anchorCount; i += 1) {
      const px = points[i * 2];
      const py = points[i * 2 + 1];
      for (let j = 0; j < coordsPerAnchor; j += 2) {
        const base = i * coordsPerAnchor + j;
        output[base] = px + distance[base];
        output[base + 1] = py + distance[base + 1];
      }
    }
    return { data: output, numKeypoints };
  }

  private concatFloat32(list: Float32Array[]) {
    if (!list.length) return null;
    const total = list.reduce((sum, arr) => sum + arr.length, 0);
    const merged = new Float32Array(total);
    let offset = 0;
    list.forEach((arr) => {
      merged.set(arr, offset);
      offset += arr.length;
    });
    return merged;
  }

  private toFloat32(tensor?: ort.Tensor, expectedLength?: number) {
    if (!tensor) return new Float32Array();
    const data =
      tensor.data instanceof Float32Array ? tensor.data : Float32Array.from(tensor.data as Iterable<number>);
    if (!expectedLength) return data;
    if (data.length === expectedLength) return data;
    if (data.length > expectedLength) {
      return data.subarray(0, expectedLength);
    }
    return data;
  }

  private extractScores(data: Float32Array, K: number) {
    if (data.length === K) {
      return data;
    }
    if (data.length === K * 2) {
      const scores = new Float32Array(K);
      for (let i = 0; i < K; i += 1) {
        scores[i] = data[i * 2 + 1];
      }
      return scores;
    }
    return data.subarray(0, K);
  }

  private nms(detections: Detection[]) {
    if (!detections.length) return detections;
    const objects = detections.map((det, index) => ({ ...det, index }));
    objects.sort((a, b) => b.score - a.score);
    const keep: typeof detections = [];
    while (objects.length) {
      const current = objects.shift()!;
      keep.push(current);
      for (let i = objects.length - 1; i >= 0; i -= 1) {
        const candidate = objects[i];
        const iou = this.iou(current.bbox, candidate.bbox);
        if (iou > this.nmsThresh) {
          objects.splice(i, 1);
        }
      }
    }
    return keep.map(({ bbox, score, kps }) => ({ bbox, score, kps }));
  }

  private iou(boxA: number[], boxB: number[]) {
    const x1 = Math.max(boxA[0], boxB[0]);
    const y1 = Math.max(boxA[1], boxB[1]);
    const x2 = Math.min(boxA[2], boxB[2]);
    const y2 = Math.min(boxA[3], boxB[3]);
    const w = Math.max(0, x2 - x1 + 1);
    const h = Math.max(0, y2 - y1 + 1);
    const inter = w * h;
    const areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
    const areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);
    const union = areaA + areaB - inter;
    return union <= 0 ? 0 : inter / union;
  }
}

class LandmarkEstimator {
  private session: ort.InferenceSession | null = null;
  private inputName: string | null = null;
  private outputNames: string[] = [];
  private inputSize: [number, number] = [112, 112];
  private cropCanvas: HTMLCanvasElement;
  private cropCtx: CanvasRenderingContext2D;
  private preprocessCanvas: HTMLCanvasElement;
  private preCtx: CanvasRenderingContext2D;

  constructor() {
    this.cropCanvas = document.createElement('canvas');
    const cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true });
    if (!cropCtx) {
      throw new Error('无法创建关键点裁剪上下文');
    }
    this.cropCtx = cropCtx;
    this.preprocessCanvas = document.createElement('canvas');
    const preCtx = this.preprocessCanvas.getContext('2d', { willReadFrequently: true });
    if (!preCtx) {
      throw new Error('无法创建关键点预处理上下文');
    }
    this.preCtx = preCtx;
  }

  isReady() {
    return Boolean(this.session);
  }

  async loadModel(source: ModelSource) {
    const buffer = await this.asArrayBuffer(source);
    this.session = await createOrtSession(buffer);
    this.inputName = this.session.inputNames[0];
    this.outputNames = this.session.outputNames;
    const inputMeta = this.session.inputMetadata[this.inputName];
    const shape = inputMeta?.dimensions ?? [1, 3, 112, 112];
    if (typeof shape[2] === 'number' && typeof shape[3] === 'number') {
      this.inputSize = [shape[3], shape[2]];
    }
  }

  private async asArrayBuffer(source: ModelSource) {
    const candidates = Array.isArray(source) ? source : [source];
    const errors: string[] = [];
    for (const entry of candidates) {
      try {
        if (entry instanceof ArrayBuffer) {
          return entry;
        }
        if (entry instanceof Blob) {
          return entry.arrayBuffer();
        }
        if (typeof entry === 'string') {
          return await fetchModelFromUrl(entry);
        }
      } catch (err) {
        errors.push(err instanceof Error ? err.message : String(err));
      }
    }
    throw new Error(errors.length ? errors.join('; ') : '不支持的模型来源');
  }

  async estimate(element: InputElement, bbox: [number, number, number, number]) {
    if (!this.session || !this.inputName) {
      throw new Error('关键点模型尚未加载');
    }
    const { tensor, meta } = this.prepareInput(element, bbox);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.inputName] = tensor;
    const result = await this.session.run(feeds);
    const { landmarkTensor, poseTensor } = this.selectOutputs(result);
    const data = this.toFloat32(landmarkTensor);
    if (!data || !data.length) {
      return null;
    }
    const poseData = this.toFloat32(poseTensor);
    return {
      landmarks: this.projectLandmarks(data, meta),
      pose: this.parsePose(poseData),
    };
  }

  private selectOutputs(result: ort.InferenceSession.OnnxValueMapType) {
    let poseTensor: ort.Tensor | null = null;
    let landmarkTensor: ort.Tensor | null = null;
    this.outputNames.forEach((name) => {
      const tensor = result[name];
      if (!tensor) return;
      const length = tensor.data?.length ?? (Array.isArray(tensor) ? tensor.length : 0);
      if (length <= 6 && !poseTensor) {
        poseTensor = tensor;
      } else if (length > 6 && !landmarkTensor) {
        landmarkTensor = tensor;
      }
    });
    if (!landmarkTensor) {
      landmarkTensor = result[this.outputNames[0]];
    }
    return { landmarkTensor, poseTensor };
  }

  private prepareInput(element: InputElement, bbox: [number, number, number, number]) {
    const width = (element as HTMLVideoElement).videoWidth || element.width;
    const height = (element as HTMLVideoElement).videoHeight || element.height;
    if (!width || !height) {
      throw new Error('输入尺寸无效');
    }
    let [x1, y1, x2, y2] = bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    const maxSide = Math.max(w, h);
    const sizeW = Math.max(1, Math.round(maxSide * 0.9));
    const sizeH = Math.max(1, Math.round(maxSide * 0.9));
    const cx = x1 + w / 2;
    const cy = y1 + h / 2;
    let cropX1 = Math.round(cx - sizeW / 2);
    let cropY1 = Math.round(cy - sizeH * 0.4);
    let cropX2 = cropX1 + sizeW;
    let cropY2 = cropY1 + sizeH;

    const leftPad = Math.max(0, -cropX1);
    const topPad = Math.max(0, -cropY1);
    const rightPad = Math.max(0, cropX2 - width);
    const bottomPad = Math.max(0, cropY2 - height);

    const drawX1 = Math.max(0, cropX1);
    const drawY1 = Math.max(0, cropY1);
    const drawX2 = Math.min(width, cropX2);
    const drawY2 = Math.min(height, cropY2);
    const drawWidth = Math.max(0, drawX2 - drawX1);
    const drawHeight = Math.max(0, drawY2 - drawY1);

    this.cropCanvas.width = sizeW;
    this.cropCanvas.height = sizeH;
    this.cropCtx.fillStyle = 'black';
    this.cropCtx.fillRect(0, 0, sizeW, sizeH);
    if (drawWidth > 0 && drawHeight > 0) {
      this.cropCtx.drawImage(
        element,
        drawX1,
        drawY1,
        drawWidth,
        drawHeight,
        leftPad,
        topPad,
        drawWidth,
        drawHeight,
      );
    }

    const [targetW, targetH] = this.inputSize;
    this.preprocessCanvas.width = targetW;
    this.preprocessCanvas.height = targetH;
    this.preCtx.drawImage(this.cropCanvas, 0, 0, sizeW, sizeH, 0, 0, targetW, targetH);
    const imageData = this.preCtx.getImageData(0, 0, targetW, targetH).data;
    const area = targetW * targetH;
    const floatData = new Float32Array(area * 3);
    for (let i = 0; i < area; i += 1) {
      const base = i * 4;
      floatData[i] = imageData[base] / 255;
      floatData[i + area] = imageData[base + 1] / 255;
      floatData[i + area * 2] = imageData[base + 2] / 255;
    }
    const tensor = new ort.Tensor('float32', floatData, [1, 3, targetH, targetW]);
    return {
      tensor,
      meta: {
        originX: drawX1 - leftPad,
        originY: drawY1 - topPad,
        sizeW,
        sizeH,
        pad: { left: leftPad, top: topPad, right: rightPad, bottom: bottomPad },
      },
    };
  }

  private toFloat32(tensor?: ort.Tensor | null) {
    if (!tensor) return null;
    if (tensor.data instanceof Float32Array) {
      return tensor.data;
    }
    if (Array.isArray(tensor.data)) {
      return Float32Array.from(tensor.data);
    }
    if (tensor instanceof Float32Array) {
      return tensor;
    }
    if (Array.isArray(tensor)) {
      return Float32Array.from(tensor);
    }
    return null;
  }

  private projectLandmarks(values: Float32Array, meta: { originX: number; originY: number; sizeW: number; sizeH: number }) {
    const coords: Array<{ x: number; y: number }> = [];
    for (let i = 0; i < values.length; i += 2) {
      const px = values[i] * meta.sizeW;
      const py = values[i + 1] * meta.sizeH;
      coords.push({
        x: meta.originX + px,
        y: meta.originY + py,
      });
    }
    return coords;
  }

  private parsePose(values: Float32Array | null): Pose | null {
    if (!values || !values.length) return null;
    return {
      yaw: values[0] || 0,
      pitch: values[1] || 0,
      roll: values[2] || 0,
    };
  }
}

function toCanvasPoint(point?: { x: number; y: number } | null) {
  if (!point || typeof point.x !== 'number' || typeof point.y !== 'number') {
    return null;
  }
  return { x: point.x, y: point.y };
}

function getFaceCenterAndNose(det: DetectionWithLandmarks | null) {
  if (!det || !det.landmarks || det.landmarks.length < 98 || !Array.isArray(det.bbox)) {
    return null;
  }
  const tl = toCanvasPoint(det.landmarks[96]);
  const tr = toCanvasPoint(det.landmarks[97]);
  const bl = toCanvasPoint(det.landmarks[76]);
  const br = toCanvasPoint(det.landmarks[82]);
  const nose = toCanvasPoint(det.landmarks[54]);
  if (!tl || !tr || !bl || !br || !nose) {
    return null;
  }
  const centerX = (tl.x + tr.x + bl.x + br.x) / 4;
  const centerY = (tl.y + tr.y + bl.y + br.y) / 4;
  const width = Math.max(1, det.bbox[2] - det.bbox[0]);
  const height = Math.max(1, det.bbox[3] - det.bbox[1]);
  return {
    centerX,
    centerY,
    noseX: nose.x,
    noseY: nose.y,
    width,
    height,
  };
}

function toPoseDegrees(pose: Pose | null) {
  if (!pose) return null;
  const radToDeg = 180 / Math.PI;
  return {
    yaw: pose.yaw * radToDeg,
    pitch: pose.pitch * radToDeg,
    roll: pose.roll * radToDeg,
  };
}

function computePoseScore(pose: Pose | null) {
  if (!pose) return null;
  const { yaw, pitch, roll } = pose;
  const dist = Math.sqrt(
    (yaw * POSE_SCORE_WEIGHTS.yaw) ** 2 +
    (pitch * POSE_SCORE_WEIGHTS.pitch) ** 2 +
    (roll * POSE_SCORE_WEIGHTS.roll) ** 2,
  );
  const score = 100 * Math.exp(-(dist ** 2) / (2 * POSE_SCORE_SIGMA ** 2));
  const rounded = Math.round(score * 100) / 100;
  return Math.max(0, Math.min(100, rounded));
}

function computeNodRatio(det: DetectionWithLandmarks | null) {
  const info = getFaceCenterAndNose(det);
  if (!info) return null;
  return (info.centerY - info.noseY) / info.height;
}

function computeShakeRatio(det: DetectionWithLandmarks | null) {
  const info = getFaceCenterAndNose(det);
  if (!info) return null;
  return (info.centerX - info.noseX) / info.width;
}

function updateRangeState(range: RangeState, ratio: number | null) {
  if (ratio === null || !Number.isFinite(ratio)) return;
  if (range.min === null || range.max === null) {
    range.min = ratio;
    range.max = ratio;
  } else {
    range.min = Math.min(range.min, ratio);
    range.max = Math.max(range.max, ratio);
  }
}

function getSpread(range: RangeState) {
  if (range.min === null || range.max === null) return null;
  return range.max - range.min;
}

function resetRange(range: RangeState) {
  range.min = null;
  range.max = null;
}

function computeBlinkRatios(det: DetectionWithLandmarks | null) {
  if (!det || !det.landmarks) return null;
  const upperLeft = toCanvasPoint(det.landmarks[66]);
  const lowerLeft = toCanvasPoint(det.landmarks[62]);
  const outerLeft = toCanvasPoint(det.landmarks[64]);
  const innerLeft = toCanvasPoint(det.landmarks[60]);
  const upperRight = toCanvasPoint(det.landmarks[74]);
  const lowerRight = toCanvasPoint(det.landmarks[70]);
  const outerRight = toCanvasPoint(det.landmarks[72]);
  const innerRight = toCanvasPoint(det.landmarks[68]);
  if (!upperLeft || !lowerLeft || !outerLeft || !innerLeft || !upperRight || !lowerRight || !outerRight || !innerRight) {
    return null;
  }
  const leftHeight = Math.abs(upperLeft.y - lowerLeft.y);
  const rightHeight = Math.abs(upperRight.y - lowerRight.y);
  const leftWidth = Math.abs(outerLeft.x - innerLeft.x);
  const rightWidth = Math.abs(outerRight.x - innerRight.x);
  const leftRatio = leftHeight === 0 ? Infinity : leftWidth / leftHeight;
  const rightRatio = rightHeight === 0 ? Infinity : rightWidth / rightHeight;
  const ratio = Math.min(leftRatio, rightRatio);
  return {
    leftHeight,
    rightHeight,
    leftRatio,
    rightRatio,
    ratio,
  };
}

function updateBlinkState(metrics: ReturnType<typeof computeBlinkRatios>, state: LivenessComputationState) {
  if (!metrics) return;
  const blinkState = state.blink;
  if (!Number.isFinite(blinkState.eyeAvg ?? NaN)) {
    blinkState.eyeAvg = INITIAL_BLINK_AVG;
  }
  const threshold = blinkState.eyeAvg as number;
  const closedDetected =
    metrics.leftHeight === 0 ||
    metrics.rightHeight === 0 ||
    (typeof metrics.ratio === 'number' && metrics.leftRatio > threshold && metrics.rightRatio > threshold);
  if (closedDetected) {
    blinkState.closedDetected = true;
  }
  const openDetected =
    metrics.leftHeight > 0 &&
    metrics.rightHeight > 0 &&
    metrics.leftRatio < threshold &&
    metrics.rightRatio < threshold;
  if (openDetected) {
    blinkState.openDetected = true;
  }
  updateBlinkAverage(metrics.ratio, state);
}

function updateBlinkAverage(ratio: number | null | undefined, state: LivenessComputationState) {
  if (typeof ratio !== 'number' || !Number.isFinite(ratio)) {
    return;
  }
  const blinkState = state.blink;
  if (!Number.isFinite(blinkState.eyeAvg ?? NaN)) {
    blinkState.eyeAvg = ratio;
    return;
  }
  if (ratio < BLINK_THRESHOLD_LIMITS.low || ratio > BLINK_THRESHOLD_LIMITS.high) {
    return;
  }
  const currentWeight = BLINK_AVG_WEIGHTS.current;
  const previousWeight = BLINK_AVG_WEIGHTS.previous;
  blinkState.eyeAvg = ratio * currentWeight + (blinkState.eyeAvg as number) * previousWeight;
}

function resetBlinkDetections(state: LivenessComputationState, options: { resetAverage?: boolean } = {}) {
  state.blink.closedDetected = false;
  state.blink.openDetected = false;
  if (options.resetAverage) {
    state.blink.eyeAvg = INITIAL_BLINK_AVG;
  }
}

function computeMouthRatio(det: DetectionWithLandmarks | null) {
  if (!det || !det.landmarks) return null;
  const upper = toCanvasPoint(det.landmarks[90]);
  const lower = toCanvasPoint(det.landmarks[94]);
  const left = toCanvasPoint(det.landmarks[88]);
  const right = toCanvasPoint(det.landmarks[92]);
  if (!upper || !lower || !left || !right) {
    return null;
  }
  const width = Math.abs(right.x - left.x);
  if (width === 0) {
    return null;
  }
  const height = Math.abs(lower.y - upper.y);
  return height / width;
}

function updateMouthState(ratio: number | null, state: LivenessComputationState) {
  if (typeof ratio !== 'number' || !Number.isFinite(ratio)) {
    return;
  }
  if (ratio > LIVENESS_CONFIG.mouthThreshold) {
    state.mouth.openDetected = true;
  } else {
    state.mouth.closedDetected = true;
  }
}

function resetMouthState(state: LivenessComputationState) {
  state.mouth.openDetected = false;
  state.mouth.closedDetected = false;
}

export interface StageStatus {
  total: number;
  stageIndex: number;
  currentStage: LivenessStageDefinition | null;
  completed: boolean;
  justCompletedStage?: LivenessStageDefinition;
  justCompletedIndex?: number;
}

export interface ProcessFrameResult {
  detection: DetectionWithLandmarks | null;
  metrics: LivenessMetrics | null;
  stage: StageStatus;
  multiFaceDetected?: boolean;
}

interface LivenessEngineOptions {
  detectorModelUrl?: string;
  landmarkModelUrl?: string;
  detectionThreshold?: number;
}

export class LivenessEngine {
  private detector = new SCRFDDetector();
  private landmarkEstimator = new LandmarkEstimator();
  private state: LivenessComputationState = createLivenessComputationState();
  private detectorModelSources: string[];
  private landmarkModelSources: string[];
  private detectionThreshold: number;
  private modelsLoaded = false;

  constructor(options: LivenessEngineOptions = {}) {
    const detectorUrl = options.detectorModelUrl ?? DEFAULT_MODEL_URL;
    const landmarkUrl = options.landmarkModelUrl ?? DEFAULT_LANDMARK_MODEL_URL;
    this.detectorModelSources = buildModelSourceCandidates(detectorUrl, SCRFD_RELATIVE_PATH);
    this.landmarkModelSources = buildModelSourceCandidates(landmarkUrl, PFPLD_RELATIVE_PATH);
    this.detectionThreshold = options.detectionThreshold ?? 0.5;
    configureOrtEnv();
    applyDeviceBackendOverrides();
    if (isSafari()) {
      forceScalarWasmBackend();
    }
  }

  async warmup() {
    if (this.modelsLoaded) return;
    await this.detector.loadModel(this.detectorModelSources);
    await this.landmarkEstimator.loadModel(this.landmarkModelSources);
    this.modelsLoaded = true;
  }

  async detectImage(element: InputElement): Promise<{ detections: FaceDetectionResult[]; primary: FaceDetectionResult | null }> {
    await this.warmup();
    const detections = await this.detector.detectFromElement(element, this.detectionThreshold);
    return { detections, primary: detections[0] ?? null };
  }

  async estimateLandmarks(element: InputElement, bbox: [number, number, number, number]) {
    await this.warmup();
    return this.landmarkEstimator.estimate(element, bbox);
  }

  startSession() {
    this.state = createLivenessComputationState();
    this.state.active = true;
    resetBlinkDetections(this.state, { resetAverage: true });
    resetMouthState(this.state);
    resetRange(this.state.nodRange);
    resetRange(this.state.shakeRange);
  }

  stopSession() {
    this.state.active = false;
  }

  isReady() {
    return this.modelsLoaded && this.detector.isReady() && this.landmarkEstimator.isReady();
  }

  getCurrentStage() {
    return LIVENESS_STAGES[this.state.stageIndex] ?? null;
  }

  async processFrame(element: InputElement): Promise<ProcessFrameResult> {
    if (!this.state.active) {
      return {
        detection: null,
        metrics: null,
        stage: this.buildStageStatus(),
      };
    }
    await this.warmup();
    const detections = await this.detector.detectFromElement(element, this.detectionThreshold);
    if (!detections.length) {
      this.state.lastMetrics = null;
      return {
        detection: null,
        metrics: null,
        stage: this.buildStageStatus(),
      };
    }
    if (detections.length > 1) {
      this.state.lastMetrics = null;
      return {
        detection: null,
        metrics: null,
        stage: this.buildStageStatus(),
        multiFaceDetected: true,
      };
    }
    const primary = detections[0];
    const estimate = await this.landmarkEstimator.estimate(element, primary.bbox);
    if (!estimate) {
      this.state.lastMetrics = null;
      return {
        detection: null,
        metrics: null,
        stage: this.buildStageStatus(),
      };
    }
    const detection: DetectionWithLandmarks = {
      ...primary,
      landmarks: estimate.landmarks,
      pose: estimate.pose,
    };
    const nodRatio = computeNodRatio(detection);
    const shakeRatio = computeShakeRatio(detection);
    updateRangeState(this.state.nodRange, nodRatio);
    updateRangeState(this.state.shakeRange, shakeRatio);
    const nodSpread = getSpread(this.state.nodRange);
    const shakeSpread = getSpread(this.state.shakeRange);
    const blinkMetrics = computeBlinkRatios(detection);
    updateBlinkState(blinkMetrics, this.state);
    const mouthRatio = computeMouthRatio(detection);
    updateMouthState(mouthRatio, this.state);
    const poseDegrees = toPoseDegrees(detection.pose);
    const frontalScore = poseDegrees ? computePoseScore(poseDegrees) : null;

    const metrics: LivenessMetrics = {
      nodRatio,
      nodSpread,
      shakeRatio,
      shakeSpread,
      blink: blinkMetrics,
      mouthRatio,
      pose: detection.pose,
      poseDegrees,
      frontalScore,
    };
    this.state.lastMetrics = metrics;
    const stageStatus = this.evaluateStage(metrics);
    return {
      detection,
      metrics,
      stage: stageStatus,
    };
  }

  private evaluateStage(metrics: LivenessMetrics): StageStatus {
    const stage = LIVENESS_STAGES[this.state.stageIndex];
    if (!stage) {
      this.state.completed = true;
      this.state.active = false;
      return this.buildStageStatus();
    }
    const completed = stage.check(metrics, this.state);
    if (!completed) {
      return this.buildStageStatus();
    }
    const completedIndex = this.state.stageIndex;
    this.state.stageIndex += 1;
    this.state.progress = this.state.stageIndex / LIVENESS_STAGES.length;
    if (stage.key === 'blink') {
      resetBlinkDetections(this.state);
    } else if (stage.key === 'mouth') {
      resetMouthState(this.state);
    } else if (stage.key === 'nod') {
      resetRange(this.state.nodRange);
    } else if (stage.key === 'shake') {
      resetRange(this.state.shakeRange);
    }
    if (this.state.stageIndex >= LIVENESS_STAGES.length) {
      this.state.completed = true;
      this.state.active = false;
    }
    return this.buildStageStatus({
      justCompletedStage: stage,
      justCompletedIndex: completedIndex,
    });
  }

  private buildStageStatus(extra?: { justCompletedStage?: LivenessStageDefinition; justCompletedIndex?: number }): StageStatus {
    const status: StageStatus = {
      total: LIVENESS_STAGES.length,
      stageIndex: Math.min(this.state.stageIndex, LIVENESS_STAGES.length - 1),
      currentStage: this.getCurrentStage(),
      completed: this.state.completed,
    };
    if (extra?.justCompletedStage) {
      status.justCompletedStage = extra.justCompletedStage;
      status.justCompletedIndex = extra.justCompletedIndex;
    }
    return status;
  }
}
