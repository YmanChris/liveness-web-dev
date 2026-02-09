
import React, { useState, useRef, useEffect, useCallback } from 'react';
import gifshot from 'gifshot';
import { CheckCircle2, Loader2, Activity, PowerOff, Eye, MoveVertical, MoveHorizontal, Check, QrCode, Smile } from 'lucide-react';
import { LivenessEngine, LIVENESS_STAGES, type LivenessStageKey, type ProcessFrameResult } from '../services/livenessEngine';

interface Challenge {
  key: LivenessStageKey;
  instruction: string;
  label: string;
  icon: React.ElementType;
}

const CHALLENGE_META: Record<LivenessStageKey, { label: string; instruction: string; icon: React.ElementType }> = {
    nod: { label: 'NOD', instruction: 'Nod Head', icon: MoveVertical },
    shake: { label: 'SHAKE', instruction: 'Shake Head', icon: MoveHorizontal },
    blink: { label: 'BLINK', instruction: 'Blink', icon: Eye },
    mouth: { label: 'OPEN MOUTH', instruction: 'Open Mouth', icon: Smile },
};

const CHALLENGES: Challenge[] = LIVENESS_STAGES.map((stage) => ({
  key: stage.key,
  label: CHALLENGE_META[stage.key].label,
  instruction: CHALLENGE_META[stage.key].instruction,
  icon: CHALLENGE_META[stage.key].icon,
}));

const DEFAULT_STAGE_TIMEOUT_MS = 3000;
const MIN_STAGE_COMPLETE_MS = 1000;
const EVIDENCE_FRAME_COUNT = 15;
const STAGE_TIMEOUTS_MS: Partial<Record<LivenessStageKey, number>> = {
  nod: DEFAULT_STAGE_TIMEOUT_MS,
  shake: DEFAULT_STAGE_TIMEOUT_MS,
  blink: DEFAULT_STAGE_TIMEOUT_MS,
  mouth: DEFAULT_STAGE_TIMEOUT_MS,
};

const FaceLiveness: React.FC = () => {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isMobileHandoff, setIsMobileHandoff] = useState(false);
  const [mobileStatus, setMobileStatus] = useState<'WAITING' | 'CONNECTED' | 'PROCESSING' | 'COMPLETED'>('WAITING');
  
  const [sessionState, setSessionState] = useState<'IDLE' | 'READY' | 'LOADING' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED'>('IDLE');
  const [currentChallengeIndex, setCurrentChallengeIndex] = useState(0);
  const [result, setResult] = useState<'PASS' | 'FAIL' | null>(null);
  const [evidenceMedia, setEvidenceMedia] = useState<string[]>(() => Array(CHALLENGES.length).fill(''));
  const [instructionText, setInstructionText] = useState("");
  const [isChallengePassing, setIsChallengePassing] = useState(false);
  const [isStageCooldown, setIsStageCooldown] = useState(false);
  const [isCompletionDelay, setIsCompletionDelay] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isEngineReady, setIsEngineReady] = useState(false);
  const [algoFps, setAlgoFps] = useState<number | null>(null);
  const [multiFaceWarning, setMultiFaceWarning] = useState(false);
  const [stageProgress, setStageProgress] = useState(1);
  const [keyFrameImage, setKeyFrameImage] = useState<string | null>(null);
  const [qualityScore, setQualityScore] = useState<number | null>(null);
  const [brightnessScore, setBrightnessScore] = useState<number | null>(null);
  const [brightnessStatus, setBrightnessStatus] = useState<string | null>(null);
  const [poseText, setPoseText] = useState('--');
  const [frontalFaceScore, setFrontalFaceScore] = useState<number | null>(null);
  const [clarityScore, setClarityScore] = useState<number | null>(null);
  const [uniformLightingScore, setUniformLightingScore] = useState<number | null>(null);
  const [backgroundUniformityScore, setBackgroundUniformityScore] = useState<number | null>(null);
  const [pixelResolutionScore, setPixelResolutionScore] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inferenceRafRef = useRef<number | null>(null);
  const engineRef = useRef<LivenessEngine | null>(null);
  const engineBusyRef = useRef(false);
  const completionFlashRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stageCooldownRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isStageCooldownRef = useRef(false);
  const completionDelayRef = useRef(false);
  const completionDelayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const completedStageKeyRef = useRef<LivenessStageKey | null>(null);
  const stageStartTimesRef = useRef<Record<LivenessStageKey, number>>({} as Record<LivenessStageKey, number>);
  const analysisTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fpsCounterRef = useRef(0);
  const fpsLastTimeRef = useRef<number | null>(null);
  const stageStartTimeRef = useRef<number | null>(null);
  const progressRafRef = useRef<number | null>(null);
  const stageTimeoutRef = useRef(DEFAULT_STAGE_TIMEOUT_MS);
  const currentStageKeyRef = useRef<LivenessStageKey | null>(null);
  const keyFrameScoreRef = useRef<number>(-Infinity);
  const keyFrameOriginalRef = useRef<string | null>(null);
  const keyFrameFullRef = useRef<string | null>(null);
  const stageFrameBufferRef = useRef<Record<LivenessStageKey, string[]>>({
    nod: [],
    shake: [],
    blink: [],
    mouth: [],
  });
  const evidenceMediaRef = useRef<string[]>(Array(CHALLENGES.length).fill(''));
  const pendingGifPromisesRef = useRef<Promise<void>[]>([]);
  const artifactsSavedRef = useRef(false);

  const buildConsolePayload = useCallback((status: 'PASS' | 'FAIL') => {
    const gifMap: Record<string, string | null> = {
      nod: evidenceMediaRef.current[0] || null,
      shake: evidenceMediaRef.current[1] || null,
      blink: evidenceMediaRef.current[2] || null,
      open: evidenceMediaRef.current[3] || null,
    };
    return {
      event: 'LIVENESS_RESULT',
      status,
      sessionState,
      scores: {
        qualityScore,
        brightnessScore,
        brightnessStatus,
        frontalFaceScore,
        clarityScore,
        uniformLightingScore,
        backgroundUniformityScore,
        pixelResolutionScore,
      },
      images: {
        faceFull: keyFrameFullRef.current || keyFrameImage,
        faceCrop: keyFrameOriginalRef.current,
        gifs: gifMap,
      },
    };
  }, [
    backgroundUniformityScore,
    brightnessScore,
    brightnessStatus,
    clarityScore,
    frontalFaceScore,
    keyFrameImage,
    pixelResolutionScore,
    qualityScore,
    sessionState,
    uniformLightingScore,
  ]);

  const logConsoleResult = useCallback((status: 'PASS' | 'FAIL') => {
    const payload = buildConsolePayload(status);
    console.log('[LIVENESS_RESULT]', JSON.stringify(payload));
  }, [buildConsolePayload]);

  useEffect(() => {
    engineRef.current = new LivenessEngine();
    return () => {
      engineRef.current?.stopSession();
    };
  }, []);

  useEffect(() => {
    return () => stopCamera();
  }, []);

  useEffect(() => {
    void startCamera();
  }, []);

  useEffect(() => {
    if (isCameraActive && sessionState === 'READY' && !isLoadingModels) {
      void runLivenessSession();
    }
  }, [isCameraActive, isLoadingModels, sessionState]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: 'user' } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsCameraActive(true);
        setIsMobileHandoff(false);
        engineRef.current?.stopSession();
        setSessionState('READY');
        setInstructionText("");
        setResult(null);
        setEvidenceMedia(Array(CHALLENGES.length).fill(''));
        setIsChallengePassing(false);
        setIsStageCooldown(false);
        setIsCompletionDelay(false);
        isStageCooldownRef.current = false;
        completionDelayRef.current = false;
        completedStageKeyRef.current = null;
        stageStartTimesRef.current = {} as Record<LivenessStageKey, number>;
        setCurrentChallengeIndex(0);
        setIsEngineReady(false);
        setIsLoadingModels(false);
        stageFrameBufferRef.current = {
          nod: [],
          shake: [],
          blink: [],
          mouth: [],
        };
        setAlgoFps(null);
        fpsCounterRef.current = 0;
        fpsLastTimeRef.current = null;
        stageStartTimeRef.current = null;
        currentStageKeyRef.current = null;
        stageTimeoutRef.current = DEFAULT_STAGE_TIMEOUT_MS;
        if (progressRafRef.current) {
          cancelAnimationFrame(progressRafRef.current);
          progressRafRef.current = null;
        }
        setMultiFaceWarning(false);
        setStageProgress(1);
        setKeyFrameImage(null);
        keyFrameScoreRef.current = -Infinity;
        keyFrameOriginalRef.current = null;
        setQualityScore(null);
        setBrightnessScore(null);
        setBrightnessStatus(null);
        setPoseText('--');
        setFrontalFaceScore(null);
        setClarityScore(null);
        setUniformLightingScore(null);
        setBackgroundUniformityScore(null);
        setPixelResolutionScore(null);
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      // Auto-fallback to mobile on error
      startMobileHandoff();
    }
  };

  const startMobileHandoff = () => {
      stopCamera(); // Ensure clean state
      setIsMobileHandoff(true);
      setMobileStatus('WAITING');
      setSessionState('IDLE');
      setResult(null);
      setEvidenceMedia(Array(CHALLENGES.length).fill(''));
      setCurrentChallengeIndex(0);
      engineRef.current?.stopSession();
      setIsEngineReady(false);
      setIsLoadingModels(false);
      stageFrameBufferRef.current = {
        nod: [],
        shake: [],
        blink: [],
        mouth: [],
      };
      
      // Simulate Mobile Flow Interaction
      setTimeout(() => {
          setMobileStatus('CONNECTED');
          setSessionState('IN_PROGRESS');
          
          setTimeout(() => {
              setMobileStatus('PROCESSING');
              // Simulate incoming evidence from mobile (using static images as fallback for demo)
              const mockFrames = Array(4).fill("https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=200&h=200&fit=crop");
              setEvidenceMedia(mockFrames);
              
              setTimeout(() => {
                   setMobileStatus('COMPLETED');
                   finishSession();
              }, 2500);
          }, 2000);
      }, 3000);
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    if (inferenceRafRef.current) {
      cancelAnimationFrame(inferenceRafRef.current);
      inferenceRafRef.current = null;
    }
    if (completionFlashRef.current) {
      clearTimeout(completionFlashRef.current);
      completionFlashRef.current = null;
    }
    if (stageCooldownRef.current) {
      clearTimeout(stageCooldownRef.current);
      stageCooldownRef.current = null;
    }
    if (completionDelayTimerRef.current) {
      clearTimeout(completionDelayTimerRef.current);
      completionDelayTimerRef.current = null;
    }
    if (analysisTimerRef.current) {
      clearTimeout(analysisTimerRef.current);
      analysisTimerRef.current = null;
    }
    engineRef.current?.stopSession();
    engineBusyRef.current = false;
    setIsCameraActive(false);
    setIsMobileHandoff(false);
    setSessionState('IDLE');
    setInstructionText("");
    setIsChallengePassing(false);
    setIsStageCooldown(false);
    setIsCompletionDelay(false);
    isStageCooldownRef.current = false;
    completionDelayRef.current = false;
    completedStageKeyRef.current = null;
    stageStartTimesRef.current = {} as Record<LivenessStageKey, number>;
    setResult(null);
    setEvidenceMedia(Array(CHALLENGES.length).fill(''));
    setCurrentChallengeIndex(0);
    setIsEngineReady(false);
    setIsLoadingModels(false);
    stageFrameBufferRef.current = {
      nod: [],
      shake: [],
      blink: [],
      mouth: [],
    };
    setAlgoFps(null);
    fpsCounterRef.current = 0;
    fpsLastTimeRef.current = null;
    stageStartTimeRef.current = null;
    currentStageKeyRef.current = null;
    stageTimeoutRef.current = DEFAULT_STAGE_TIMEOUT_MS;
    if (progressRafRef.current) {
      cancelAnimationFrame(progressRafRef.current);
      progressRafRef.current = null;
    }
    setMultiFaceWarning(false);
    setStageProgress(1);
    setKeyFrameImage(null);
    keyFrameScoreRef.current = -Infinity;
    setClarityScore(null);
    setUniformLightingScore(null);
    setBackgroundUniformityScore(null);
    setPixelResolutionScore(null);
    setQualityScore(null);
    setBrightnessScore(null);
    setBrightnessStatus(null);
    setFrontalFaceScore(null);
    setPoseText('--');
    keyFrameOriginalRef.current = null;
  };

  const captureFrame = useCallback(() => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = 480;
      canvas.height = 270;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.save();
        ctx.scale(-1, 1); 
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
        ctx.restore();
        return canvas.toDataURL('image/jpeg', 0.8);
      }
    }
    return null;
  }, []);

  const createEvidenceGif = useCallback((frames: string[]) => new Promise<string | null>((resolve) => {
    if (!frames.length) {
      resolve(null);
      return;
    }
    const safeFrames = frames.length >= 2 ? frames : [frames[0], frames[0], frames[0]];
    gifshot.createGIF(
      {
        images: safeFrames,
        interval: 0.08,
        gifWidth: 480,
        gifHeight: 270,
      },
      (obj: { error?: boolean; image?: string }) => {
        if (obj.error || !obj.image) {
          resolve(null);
          return;
        }
        resolve(obj.image);
      },
    );
  }), []);

  const appendStageFrame = useCallback((stageKey: LivenessStageKey) => {
    const frame = captureFrame();
    if (!frame) return;
    const buffer = stageFrameBufferRef.current[stageKey] ?? [];
    buffer.push(frame);
    if (buffer.length > EVIDENCE_FRAME_COUNT) {
      buffer.splice(0, buffer.length - EVIDENCE_FRAME_COUNT);
    }
    stageFrameBufferRef.current[stageKey] = buffer;
  }, [captureFrame]);

  const captureKeyFrame = useCallback((bbox: [number, number, number, number]) => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    if (!videoWidth || !videoHeight) return null;
    const [x1, y1, x2, y2] = bbox;
    const padding = 20;
    const cropX = Math.max(0, Math.floor(x1 - padding));
    const cropY = Math.max(0, Math.floor(y1 - padding));
    const cropW = Math.min(videoWidth - cropX, Math.ceil(x2 - x1 + padding * 2));
    const cropH = Math.min(videoHeight - cropY, Math.ceil(y2 - y1 + padding * 2));
    if (cropW <= 0 || cropH <= 0) return null;
    const originalCanvas = document.createElement('canvas');
    originalCanvas.width = cropW;
    originalCanvas.height = cropH;
    const originalCtx = originalCanvas.getContext('2d');
    if (!originalCtx) return null;
    originalCtx.save();
    originalCtx.scale(-1, 1);
    originalCtx.drawImage(video, cropX, cropY, cropW, cropH, -cropW, 0, cropW, cropH);
    originalCtx.restore();
    const scale = 64 / Math.max(cropW, cropH);
    const targetW = Math.max(1, Math.round(cropW * scale));
    const targetH = Math.max(1, Math.round(cropH * scale));
    const previewCanvas = document.createElement('canvas');
    previewCanvas.width = targetW;
    previewCanvas.height = targetH;
    const previewCtx = previewCanvas.getContext('2d');
    if (!previewCtx) return null;
    previewCtx.drawImage(originalCanvas, 0, 0, cropW, cropH, 0, 0, targetW, targetH);
    const preview = previewCanvas.toDataURL('image/jpeg', 0.85);
    return { preview };
  }, []);

  const captureUploadFrame = useCallback((bbox: [number, number, number, number]) => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;
    const [x1, y1, x2, y2] = bbox;
    const faceW = Math.max(1, x2 - x1);
    const faceH = Math.max(1, y2 - y1);
    const centerX = x1 + faceW / 2;
    const centerY = y1 + faceH / 2;
    const squareSide = Math.max(faceW, faceH);
    const expandedSide = squareSide * 1.4;
    let cropX = centerX - expandedSide / 2;
    let cropY = centerY - expandedSide / 2;
    let cropW = expandedSide;
    let cropH = expandedSide;
    cropX = Math.max(0, cropX);
    cropY = Math.max(0, cropY);
    cropW = Math.min(width - cropX, cropW);
    cropH = Math.min(height - cropY, cropH);
    if (cropW <= 0 || cropH <= 0) return null;
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(cropW);
    canvas.height = Math.round(cropH);
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(
      video,
      cropX,
      cropY,
      cropW,
      cropH,
      -canvas.width,
      0,
      canvas.width,
      canvas.height,
    );
    ctx.restore();
    return canvas.toDataURL('image/png');
  }, []);

  const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : (v > hi ? hi : v));

  const captureFullFrame = useCallback(() => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, width, height, -width, 0, width, height);
    ctx.restore();
    return canvas.toDataURL('image/png');
  }, []);

  const computeLaplacianScore = useCallback((bbox: [number, number, number, number]) => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    if (!videoWidth || !videoHeight) return null;
    const [x1, y1, x2, y2] = bbox;
    const cropX = Math.max(0, Math.floor(x1));
    const cropY = Math.max(0, Math.floor(y1));
    const cropW = Math.min(videoWidth - cropX, Math.ceil(x2 - x1));
    const cropH = Math.min(videoHeight - cropY, Math.ceil(y2 - y1));
    if (cropW <= 1 || cropH <= 1) return null;
    const canvas = document.createElement('canvas');
    canvas.width = cropW;
    canvas.height = cropH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, cropX, cropY, cropW, cropH, -cropW, 0, cropW, cropH);
    ctx.restore();
    const imageData = ctx.getImageData(0, 0, cropW, cropH);
    const gray = new Uint8Array(cropW * cropH);
    const data = imageData.data;
    for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    const k4 = [0, 1, 0, 1, -4, 1, 0, 1, 0];
    let mean = 0;
    let M2 = 0;
    let count = 0;
    const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : (v > hi ? hi : v));
    for (let y = 0; y < cropH; y += 1) {
      for (let x = 0; x < cropW; x += 1) {
        let acc = 0;
        let idx = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          const yy = clamp(y + dy, 0, cropH - 1);
          for (let dx = -1; dx <= 1; dx += 1) {
            const xx = clamp(x + dx, 0, cropW - 1);
            acc += gray[yy * cropW + xx] * k4[idx++];
          }
        }
        const v = acc;
        count += 1;
        const delta = v - mean;
        mean += delta / count;
        const delta2 = v - mean;
        M2 += delta * delta2;
      }
    }
    const variance = count > 0 ? (M2 / count) : 0;
    const sigma = Math.sqrt(Math.max(0, variance));
    const mid = 35;
    const steep = 0.08;
    const score = 100 / (1 + Math.exp(-steep * (sigma - mid)));
    return Math.round(Math.min(100, Math.max(0, score)));
  }, []);

  const computeBrightnessScore = useCallback((bbox: [number, number, number, number]) => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    if (!videoWidth || !videoHeight) return null;
    const [x1, y1, x2, y2] = bbox;
    const cropX = Math.max(0, Math.floor(x1));
    const cropY = Math.max(0, Math.floor(y1));
    const cropW = Math.min(videoWidth - cropX, Math.ceil(x2 - x1));
    const cropH = Math.min(videoHeight - cropY, Math.ceil(y2 - y1));
    if (cropW <= 1 || cropH <= 1) return null;
    const canvas = document.createElement('canvas');
    canvas.width = cropW;
    canvas.height = cropH;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, cropX, cropY, cropW, cropH, -cropW, 0, cropW, cropH);
    ctx.restore();
    const imageData = ctx.getImageData(0, 0, cropW, cropH);
    const gray = new Uint8Array(cropW * cropH);
    const data = imageData.data;
    for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    const values = Array.from(gray);
    values.sort((a, b) => a - b);
    const total = values.length;
    if (!total) return null;
    const clipPercent = 5;
    const lowIndex = Math.floor((clipPercent / 100) * (total - 1));
    const highIndex = Math.floor(((100 - clipPercent) / 100) * (total - 1));
    const pLow = values[lowIndex];
    const pHigh = values[highIndex];
    let sum = 0;
    let count = 0;
    let overexposedCount = 0;
    const overexposeGray = 240;
    for (let i = 0; i < gray.length; i += 1) {
      const v = gray[i];
      if (v >= pLow && v <= pHigh) {
        sum += v;
        count += 1;
      }
      if (v >= overexposeGray) {
        overexposedCount += 1;
      }
    }
    if (!count) return { value: 0, status: 'invalid' };
    const meanGray = sum / count;
    const brightness = (meanGray / 255) * 100;
    const overexposedRatio = overexposedCount / gray.length;
    const lowThreshold = 30;
    const highThreshold = 70;
    const overexposeRatioThresh = 0.1;
    let status = 'normal';
    if (brightness < lowThreshold) {
      status = 'underexposed';
    } else if (brightness > highThreshold && overexposedRatio > overexposeRatioThresh) {
      status = 'overexposed';
    } else if (brightness > highThreshold) {
      status = 'too bright';
    }
    const quality = Math.max(0, 100 - Math.abs(brightness - 50) * 2);
    const rounded = Math.round(quality * 100) / 100;
    return { value: rounded, status };
  }, []);

  const estimateBlurScore = useCallback((bbox: [number, number, number, number]) => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;
    const [x1, y1, x2, y2] = bbox;
    const faceW = x2 - x1;
    const faceH = y2 - y1;
    const padX = faceW * 0.2;
    const padBottom = faceH * 0.2;
    const sx = Math.max(0, Math.floor(x1 - padX));
    const sy = Math.max(0, Math.floor(y1));
    const sw = Math.min(width - sx, Math.ceil(faceW + padX * 2));
    const sh = Math.min(height - sy, Math.ceil(faceH + padBottom));
    if (sw <= 1 || sh <= 1) return null;
    const size = 200;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, sw, sh, -size, 0, size, size);
    ctx.restore();
    const imageData = ctx.getImageData(0, 0, size, size);
    const gray = new Uint8Array(size * size);
    const data = imageData.data;
    for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    let mean = 0;
    let M2 = 0;
    let count = 0;
    const k4 = [0, 1, 0, 1, -4, 1, 0, 1, 0];
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        let acc = 0;
        let idx = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          const yy = clamp(y + dy, 0, size - 1);
          for (let dx = -1; dx <= 1; dx += 1) {
            const xx = clamp(x + dx, 0, size - 1);
            acc += gray[yy * size + xx] * k4[idx++];
          }
        }
        const v = acc;
        count += 1;
        const delta = v - mean;
        mean += delta / count;
        const delta2 = v - mean;
        M2 += delta * delta2;
      }
    }
    const variance = count > 0 ? (M2 / count) : 0;
    const maxThreshold = 500;
    const score = Math.min((variance / maxThreshold) * 100, 100);
    return Math.round(score * 100) / 100;
  }, []);

  const detectUniformLighting = useCallback((bbox: [number, number, number, number]) => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;
    const [x1, y1, x2, y2] = bbox;
    const sx = Math.max(0, Math.floor(x1));
    const sy = Math.max(0, Math.floor(y1));
    const sw = Math.min(width - sx, Math.ceil(x2 - x1));
    const sh = Math.min(height - sy, Math.ceil(y2 - y1));
    if (sw <= 1 || sh <= 1) return null;
    const canvas = document.createElement('canvas');
    canvas.width = sw;
    canvas.height = sh;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, sw, sh, -sw, 0, sw, sh);
    ctx.restore();
    const imageData = ctx.getImageData(0, 0, sw, sh);
    const gray = new Uint8Array(sw * sh);
    const data = imageData.data;
    for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    const marginW = Math.floor(sw * 0.2);
    const marginH = Math.floor(sh * 0.1);
    const coreW = Math.max(1, sw - marginW * 2);
    const coreH = Math.max(1, sh - marginH * 2);
    const mid = Math.floor(coreW / 2);
    let leftSum = 0;
    let rightSum = 0;
    let leftCount = 0;
    let rightCount = 0;
    for (let y = 0; y < coreH; y += 1) {
      const yy = y + marginH;
      for (let x = 0; x < coreW; x += 1) {
        const xx = x + marginW;
        const v = gray[yy * sw + xx];
        if (x < mid) {
          leftSum += v;
          leftCount += 1;
        } else {
          rightSum += v;
          rightCount += 1;
        }
      }
    }
    const lMean = leftCount ? leftSum / leftCount : 0;
    const rMean = rightCount ? rightSum / rightCount : 0;
    const lPct = (lMean / 255) * 100;
    const rPct = (rMean / 255) * 100;
    const diffRatio = Math.abs(lPct - rPct) / Math.max(lPct, rPct, 1);
    const diffScore = diffRatio * 100;
    const score = Math.max(0, Math.min(100, 100 - diffScore));
    return Math.round(score * 100) / 100;
  }, []);

  const analyzeBackgroundUniformity = useCallback((bbox: [number, number, number, number]) => {
    if (!videoRef.current) return null;
    const video = videoRef.current;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.filter = 'blur(10px)';
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, width, height, -width, 0, width, height);
    ctx.restore();
    const imageData = ctx.getImageData(0, 0, width, height);
    const gray = new Uint8Array(width * height);
    const data = imageData.data;
    for (let i = 0, j = 0; i < data.length; i += 4, j += 1) {
      gray[j] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    const mask = new Uint8Array(width * height);
    mask.fill(1);
    const [x1, y1, x2, y2] = bbox;
    const padRatio = 0.2;
    const boxW = x2 - x1;
    const boxH = y2 - y1;
    const padW = boxW * padRatio;
    const padH = boxH * padRatio;
    const rx1 = Math.max(0, Math.floor(x1 - padW));
    const ry1 = Math.max(0, Math.floor(y1 - padH));
    const rx2 = Math.min(width, Math.ceil(x2 + padW));
    const ry2 = Math.min(height, Math.ceil(y2 + padH));
    for (let y = ry1; y < ry2; y += 1) {
      for (let x = rx1; x < rx2; x += 1) {
        mask[y * width + x] = 0;
      }
    }
    let mean = 0;
    let M2 = 0;
    let count = 0;
    for (let i = 0; i < gray.length; i += 1) {
      if (mask[i] === 0) continue;
      const v = gray[i];
      count += 1;
      const delta = v - mean;
      mean += delta / count;
      const delta2 = v - mean;
      M2 += delta * delta2;
    }
    if (!count) return 0;
    const variance = M2 / count;
    const stdDev = Math.sqrt(variance);
    const score = Math.max(0, Math.min(100, 100 - stdDev * 0.8));
    return Math.round(score * 100) / 100;
  }, []);

  const evaluatePixelResolution = useCallback((kps?: Array<{ x: number; y: number }>) => {
    if (!kps || kps.length < 2) return null;
    const left = kps[0];
    const right = kps[1];
    const dx = left.x - right.x;
    const dy = left.y - right.y;
    const iod = Math.hypot(dx, dy);
    const minIod = 80;
    const highIod = 300;
    let score = 0;
    if (iod < minIod) {
      score = (iod / minIod) * 60;
    } else if (iod >= highIod) {
      score = 100;
    } else {
      const ratio = (iod - minIod) / (highIod - minIod);
      score = 60 + ratio * 40;
    }
    return Math.round(score * 100) / 100;
  }, []);

  const handleSaveKeyFrame = useCallback(async () => {
    const fullFrame = keyFrameFullRef.current || keyFrameImage;
    if (!fullFrame) return;
    const response = await fetch(fullFrame);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `key-frame-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [keyFrameImage]);

  const downloadDataUrl = useCallback(async (dataUrl: string, fileName: string) => {
    const response = await fetch(dataUrl);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, []);

  const saveAllArtifacts = useCallback(async () => {
    if (artifactsSavedRef.current) return;
    artifactsSavedRef.current = true;
    const pending = pendingGifPromisesRef.current;
    pendingGifPromisesRef.current = [];
    if (pending.length) {
      await Promise.allSettled(pending);
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
    const timestamp = Date.now();
    const gifMap: Record<string, string | null> = {
      nod: evidenceMediaRef.current[0] || null,
      shake: evidenceMediaRef.current[1] || null,
      blink: evidenceMediaRef.current[2] || null,
      open: evidenceMediaRef.current[3] || null,
    };
    for (const [key, value] of Object.entries(gifMap)) {
      if (value) {
        await downloadDataUrl(value, `${key}-${timestamp}.gif`);
      }
    }
    const faceDataUrl = keyFrameOriginalRef.current || keyFrameFullRef.current || keyFrameImage;
    if (faceDataUrl) {
      await downloadDataUrl(faceDataUrl, `face-${timestamp}.png`);
    }
  }, [downloadDataUrl, keyFrameImage]);


  useEffect(() => {
    const values = [
      brightnessScore,
      frontalFaceScore,
      clarityScore,
      uniformLightingScore,
      backgroundUniformityScore,
      pixelResolutionScore,
    ];
    if (values.some((value) => typeof value !== 'number')) {
      return;
    }
    const sum = values.reduce((acc, value) => acc + (value as number), 0);
    const avg = sum / values.length;
    setQualityScore(Math.round(avg * 100) / 100);
  }, [
    backgroundUniformityScore,
    brightnessScore,
    clarityScore,
    frontalFaceScore,
    pixelResolutionScore,
    uniformLightingScore,
  ]);


  const finishSession = useCallback(() => {
    if (sessionState === 'COMPLETED' || sessionState === 'FAILED') return;
    engineRef.current?.stopSession();
    setInstructionText("Analyzing biometric data...");
    setIsChallengePassing(false);
    setIsStageCooldown(false);
    setIsCompletionDelay(false);
    isStageCooldownRef.current = false;
    completionDelayRef.current = false;
    completedStageKeyRef.current = null;
    stageStartTimesRef.current = {} as Record<LivenessStageKey, number>;
    setIsEngineReady(false);
    setIsLoadingModels(false);
    stageStartTimeRef.current = null;
    currentStageKeyRef.current = null;
    stageTimeoutRef.current = DEFAULT_STAGE_TIMEOUT_MS;
    if (progressRafRef.current) {
      cancelAnimationFrame(progressRafRef.current);
      progressRafRef.current = null;
    }
    setMultiFaceWarning(false);
    setStageProgress(1);
    if (completionFlashRef.current) {
      clearTimeout(completionFlashRef.current);
      completionFlashRef.current = null;
    }
    if (stageCooldownRef.current) {
      clearTimeout(stageCooldownRef.current);
      stageCooldownRef.current = null;
    }
    if (completionDelayTimerRef.current) {
      clearTimeout(completionDelayTimerRef.current);
      completionDelayTimerRef.current = null;
    }
    if (analysisTimerRef.current) {
      clearTimeout(analysisTimerRef.current);
      analysisTimerRef.current = null;
    }
    setSessionState('COMPLETED');
    setInstructionText("Verification Passed");
    setResult('PASS');
    void saveAllArtifacts().then(() => {
      logConsoleResult('PASS');
    });
  }, [logConsoleResult, saveAllArtifacts, sessionState]);

  const failSession = useCallback(() => {
    if (sessionState === 'COMPLETED' || sessionState === 'FAILED') return;
    engineRef.current?.stopSession();
    engineBusyRef.current = false;
    setInstructionText("Verification Failed");
    setIsChallengePassing(false);
    setIsStageCooldown(false);
    setIsCompletionDelay(false);
    isStageCooldownRef.current = false;
    completionDelayRef.current = false;
    completedStageKeyRef.current = null;
    stageStartTimesRef.current = {} as Record<LivenessStageKey, number>;
    setIsEngineReady(false);
    setIsLoadingModels(false);
    setResult('FAIL');
    setSessionState('FAILED');
    stageStartTimeRef.current = null;
    currentStageKeyRef.current = null;
    stageTimeoutRef.current = DEFAULT_STAGE_TIMEOUT_MS;
    if (progressRafRef.current) {
      cancelAnimationFrame(progressRafRef.current);
      progressRafRef.current = null;
    }
    setMultiFaceWarning(false);
    setStageProgress(0);
    setKeyFrameImage(null);
    keyFrameScoreRef.current = -Infinity;
    keyFrameOriginalRef.current = null;
    setQualityScore(null);
    setBrightnessScore(null);
    setBrightnessStatus(null);
    setPoseText('--');
    setFrontalFaceScore(null);
    setClarityScore(null);
    setUniformLightingScore(null);
    setBackgroundUniformityScore(null);
    setPixelResolutionScore(null);
    if (completionFlashRef.current) {
      clearTimeout(completionFlashRef.current);
      completionFlashRef.current = null;
    }
    if (stageCooldownRef.current) {
      clearTimeout(stageCooldownRef.current);
      stageCooldownRef.current = null;
    }
    if (completionDelayTimerRef.current) {
      clearTimeout(completionDelayTimerRef.current);
      completionDelayTimerRef.current = null;
    }
    if (analysisTimerRef.current) {
      clearTimeout(analysisTimerRef.current);
      analysisTimerRef.current = null;
    }
    void saveAllArtifacts().then(() => {
      logConsoleResult('FAIL');
    });
  }, [logConsoleResult, saveAllArtifacts, sessionState]);

  const resetStageTimer = useCallback((stageKey: LivenessStageKey | null) => {
    currentStageKeyRef.current = stageKey;
    stageTimeoutRef.current = stageKey ? (STAGE_TIMEOUTS_MS[stageKey] ?? DEFAULT_STAGE_TIMEOUT_MS) : DEFAULT_STAGE_TIMEOUT_MS;
    stageStartTimeRef.current = performance.now();
    setStageProgress(1);
    if (stageKey) {
      stageStartTimesRef.current[stageKey] = stageStartTimeRef.current;
      stageFrameBufferRef.current[stageKey] = [];
    }
  }, []);

  useEffect(() => {
    if (sessionState !== 'IN_PROGRESS') {
      if (progressRafRef.current) {
        cancelAnimationFrame(progressRafRef.current);
        progressRafRef.current = null;
      }
      return;
    }
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      if (isStageCooldownRef.current || completionDelayRef.current) {
        progressRafRef.current = requestAnimationFrame(tick);
        return;
      }
      const now = performance.now();
      if (stageStartTimeRef.current === null) {
        stageStartTimeRef.current = now;
        setStageProgress(1);
      }
      const elapsed = now - (stageStartTimeRef.current ?? now);
      const timeoutMs = stageTimeoutRef.current || DEFAULT_STAGE_TIMEOUT_MS;
      const progress = Math.max(0, 1 - (elapsed / timeoutMs));
      setStageProgress(progress);
      if (elapsed >= timeoutMs) {
        setStageProgress(0);
        failSession();
        return;
      }
      progressRafRef.current = requestAnimationFrame(tick);
    };
    progressRafRef.current = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      if (progressRafRef.current) {
        cancelAnimationFrame(progressRafRef.current);
        progressRafRef.current = null;
      }
    };
  }, [failSession, sessionState]);

  const triggerCompletionFlash = useCallback(() => {
    setIsChallengePassing(true);
    if (completionFlashRef.current) {
      clearTimeout(completionFlashRef.current);
    }
    completionFlashRef.current = setTimeout(() => {
      setIsChallengePassing(false);
      completionFlashRef.current = null;
    }, 800);
  }, []);

  const startStageCooldown = useCallback((nextStageKey: LivenessStageKey | null) => {
    if (nextStageKey) {
      const nextIndex = CHALLENGES.findIndex((challenge) => challenge.key === nextStageKey);
      if (nextIndex >= 0) {
        setCurrentChallengeIndex(nextIndex);
      }
    }
    isStageCooldownRef.current = true;
    setIsStageCooldown(true);
    setIsChallengePassing(true);
    stageStartTimeRef.current = null;
    setStageProgress(1);
    if (stageCooldownRef.current) {
      clearTimeout(stageCooldownRef.current);
    }
    stageCooldownRef.current = setTimeout(() => {
      isStageCooldownRef.current = false;
      setIsStageCooldown(false);
      setIsChallengePassing(false);
      stageCooldownRef.current = null;
      resetStageTimer(nextStageKey);
    }, 1000);
  }, [resetStageTimer]);

  const handleStageCompleted = useCallback(async (stageIndex: number) => {
    triggerCompletionFlash();
    const stageKey = CHALLENGES[stageIndex]?.key;
    let buffer = stageKey ? (stageFrameBufferRef.current[stageKey] ?? []) : [];
    if (buffer.length === 0) {
      const frame = captureFrame();
      if (frame) {
        buffer = [frame];
      }
    }
    const gifPromise = createEvidenceGif(buffer).then((gif) => {
      const fallback = gif ? null : captureFrame();
      const evidence = gif ?? fallback;
      if (evidence) {
        evidenceMediaRef.current[stageIndex] = evidence;
        setEvidenceMedia(prev => {
          const next = [...prev];
          next[stageIndex] = evidence;
          return next;
        });
      }
    });
    pendingGifPromisesRef.current.push(gifPromise);
    if (stageKey) {
      stageFrameBufferRef.current[stageKey] = [];
    }
  }, [captureFrame, createEvidenceGif, triggerCompletionFlash]);

  const handleLivenessResult = useCallback((result: ProcessFrameResult) => {
    const { stage, multiFaceDetected, metrics, detection } = result;
    setMultiFaceWarning(Boolean(multiFaceDetected));
    if (typeof metrics?.frontalScore === 'number') {
      setFrontalFaceScore(metrics.frontalScore);
      setPoseText(metrics.frontalScore.toFixed(2));
    }
    if (!isMobileHandoff && detection && typeof metrics?.frontalScore === 'number') {
      const poseScore = metrics.frontalScore;
      if (poseScore > keyFrameScoreRef.current) {
        const frame = captureKeyFrame(detection.bbox);
        const uploadFrame = captureUploadFrame(detection.bbox);
        const fullFrame = captureFullFrame();
        const brightness = computeBrightnessScore(detection.bbox);
        const clarity = estimateBlurScore(detection.bbox);
        const uniformLighting = detectUniformLighting(detection.bbox);
        const backgroundUniformity = analyzeBackgroundUniformity(detection.bbox);
        const pixelResolution = evaluatePixelResolution(detection.kps);
        if (frame && uploadFrame && fullFrame) {
          keyFrameScoreRef.current = poseScore;
          keyFrameOriginalRef.current = uploadFrame;
          keyFrameFullRef.current = fullFrame;
          setKeyFrameImage(fullFrame);
        }
        if (brightness) {
          setBrightnessScore(brightness.value);
          setBrightnessStatus(brightness.status);
        }
        if (typeof clarity === 'number') {
          setClarityScore(clarity);
        }
        if (typeof uniformLighting === 'number') {
          setUniformLightingScore(uniformLighting);
        }
        if (typeof backgroundUniformity === 'number') {
          setBackgroundUniformityScore(backgroundUniformity);
        }
        if (typeof pixelResolution === 'number') {
          setPixelResolutionScore(pixelResolution);
        }
      }
    }
    if (multiFaceDetected) {
      return;
    }
    if (isStageCooldownRef.current && !stage.completed) {
      return;
    }
    if (completionDelayRef.current) {
      const pendingKey = completedStageKeyRef.current;
      if (pendingKey) {
        appendStageFrame(pendingKey);
      }
      return;
    }
    const activeStage = stage.currentStage;
    if (typeof stage.justCompletedIndex === 'number') {
      const completedKey = CHALLENGES[stage.justCompletedIndex]?.key ?? null;
      const startedAt = completedKey ? stageStartTimesRef.current[completedKey] : null;
      const elapsed = performance.now() - (startedAt ?? performance.now());
      if (elapsed < MIN_STAGE_COMPLETE_MS) {
        completionDelayRef.current = true;
        setIsCompletionDelay(true);
        completedStageKeyRef.current = completedKey;
        if (completedStageKeyRef.current) {
          appendStageFrame(completedStageKeyRef.current);
        }
        const remaining = MIN_STAGE_COMPLETE_MS - elapsed;
        if (completionDelayTimerRef.current) {
          clearTimeout(completionDelayTimerRef.current);
        }
        completionDelayTimerRef.current = setTimeout(() => {
          completionDelayRef.current = false;
          setIsCompletionDelay(false);
          completionDelayTimerRef.current = null;
          completedStageKeyRef.current = null;
          void handleStageCompleted(stage.justCompletedIndex as number);
          if (stage.completed) {
            if (!analysisTimerRef.current) {
              finishSession();
            }
          } else {
            startStageCooldown(stage.currentStage?.key ?? null);
          }
        }, Math.max(0, remaining));
        return;
      }
      void handleStageCompleted(stage.justCompletedIndex);
      if (!stage.completed) {
        startStageCooldown(stage.currentStage?.key ?? null);
        return;
      }
      resetStageTimer(stage.currentStage?.key ?? null);
    }
    if (activeStage && activeStage.key !== currentStageKeyRef.current) {
      resetStageTimer(activeStage.key);
    } else if (sessionState === 'IN_PROGRESS' && stageStartTimeRef.current === null) {
      resetStageTimer(activeStage?.key ?? null);
    }
    if (activeStage) {
      appendStageFrame(activeStage.key);
    }
    if (stage.completed) {
      if (!analysisTimerRef.current) {
        finishSession();
      }
      return;
    }
    if (activeStage) {
      const idx = CHALLENGES.findIndex((challenge) => challenge.key === activeStage.key);
      if (idx >= 0) {
        setCurrentChallengeIndex(idx);
      }
    }
  }, [analyzeBackgroundUniformity, captureFullFrame, captureKeyFrame, captureUploadFrame, computeBrightnessScore, detectUniformLighting, estimateBlurScore, evaluatePixelResolution, failSession, finishSession, handleStageCompleted, isMobileHandoff, resetStageTimer, sessionState]);

  useEffect(() => {
    if (!isCameraActive) return;
    let cancelled = false;
    const loop = () => {
      if (cancelled) return;
      if (
        sessionState !== 'IN_PROGRESS' ||
        !isEngineReady ||
        !videoRef.current ||
        !engineRef.current
      ) {
        inferenceRafRef.current = requestAnimationFrame(loop);
        return;
      }
      if (completionDelayRef.current || isStageCooldownRef.current) {
        const pendingKey = completedStageKeyRef.current;
        if (pendingKey) {
          appendStageFrame(pendingKey);
        }
        inferenceRafRef.current = requestAnimationFrame(loop);
        return;
      }
      if (engineBusyRef.current) {
        inferenceRafRef.current = requestAnimationFrame(loop);
        return;
      }
      engineBusyRef.current = true;
      engineRef.current.processFrame(videoRef.current)
        .then(handleLivenessResult)
        .catch((err) => console.error('Liveness inference failed', err))
        .finally(() => {
          const now = performance.now();
          if (fpsLastTimeRef.current === null) {
            fpsLastTimeRef.current = now;
          }
          fpsCounterRef.current += 1;
          const elapsed = now - (fpsLastTimeRef.current ?? now);
          if (elapsed >= 1000) {
            const fps = fpsCounterRef.current * 1000 / elapsed;
            setAlgoFps(Math.round(fps));
            fpsCounterRef.current = 0;
            fpsLastTimeRef.current = now;
          }
          engineBusyRef.current = false;
          if (!cancelled) {
            inferenceRafRef.current = requestAnimationFrame(loop);
          }
        });
    };

    inferenceRafRef.current = requestAnimationFrame(loop);
    return () => {
      cancelled = true;
      if (inferenceRafRef.current) {
        cancelAnimationFrame(inferenceRafRef.current);
        inferenceRafRef.current = null;
      }
    };
  }, [handleLivenessResult, isCameraActive, isEngineReady, sessionState]);

  const runLivenessSession = async () => {
    if (!isCameraActive || !engineRef.current) return;
    setResult(null);
    setEvidenceMedia(Array(CHALLENGES.length).fill(''));
    evidenceMediaRef.current = Array(CHALLENGES.length).fill('');
    pendingGifPromisesRef.current = [];
    artifactsSavedRef.current = false;
    setCurrentChallengeIndex(0);
    setIsChallengePassing(false);
    setIsStageCooldown(false);
    setIsCompletionDelay(false);
    isStageCooldownRef.current = false;
    completionDelayRef.current = false;
    completedStageKeyRef.current = null;
    stageStartTimesRef.current = {} as Record<LivenessStageKey, number>;
    setInstructionText("Please Waiting");
    setIsEngineReady(false);
    setIsLoadingModels(true);
    setAlgoFps(null);
    fpsCounterRef.current = 0;
    fpsLastTimeRef.current = null;
    setMultiFaceWarning(false);
    setStageProgress(1);
    setKeyFrameImage(null);
    keyFrameScoreRef.current = -Infinity;
    stageFrameBufferRef.current = {
      nod: [],
      shake: [],
      blink: [],
      mouth: [],
    };
    engineRef.current.stopSession();
    if (stageCooldownRef.current) {
      clearTimeout(stageCooldownRef.current);
      stageCooldownRef.current = null;
    }
    if (completionDelayTimerRef.current) {
      clearTimeout(completionDelayTimerRef.current);
      completionDelayTimerRef.current = null;
    }
    if (analysisTimerRef.current) {
      clearTimeout(analysisTimerRef.current);
      analysisTimerRef.current = null;
    }
    setSessionState('LOADING');
    try {
      await engineRef.current.warmup();
      engineRef.current.startSession();
      setSessionState('IN_PROGRESS');
      setInstructionText("Nod Head");
      setIsEngineReady(true);
      setCurrentChallengeIndex(0);
      resetStageTimer(CHALLENGES[0]?.key ?? null);
    } catch (err) {
      console.error('Failed to initialize liveness session', err);
      const reason = err instanceof Error ? err.message : String(err);
      setInstructionText(`Unable to load models (${reason}). Please ensure ./models contains the ONNX files.`);
      setSessionState('READY');
      engineRef.current.stopSession();
      setIsEngineReady(false);
    }
    setIsLoadingModels(false);
  };

  const progressColor = stageProgress > 0.66
    ? 'bg-emerald-500'
    : stageProgress > 0.33
      ? 'bg-orange-500'
      : 'bg-rose-500';

  const maskStyle = {
    WebkitMaskImage: 'radial-gradient(ellipse 130px 165px at 50% 45%, transparent 98%, black 100%)',
    maskImage: 'radial-gradient(ellipse 130px 165px at 50% 45%, transparent 98%, black 100%)',
  } as const;

  return (
    <div className="h-[calc(100vh-10rem)]">
      <div className="h-full">
        <div className="bg-white border border-slate-200 rounded-2xl p-4 h-full flex flex-col shadow-sm relative overflow-hidden">
            <div className="flex justify-end items-center mb-4 px-2" />
            
            <div className="flex-1 bg-slate-100 rounded-xl relative overflow-hidden flex items-center justify-center group border border-slate-100">
               <canvas ref={canvasRef} className="hidden" />
               <div className="w-full max-w-sm aspect-[9/16] rounded-3xl overflow-hidden bg-black shadow-xl">
                 <video 
                   ref={videoRef} 
                   autoPlay 
                   playsInline 
                   muted 
                   className={`w-full h-full object-cover transform scale-x-[-1] transition-opacity duration-500 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`} 
                 />
               </div>

               {/* --- MOBILE HANDOFF UI --- */}
               {isMobileHandoff && (
                   <div className="absolute inset-0 bg-white/90 backdrop-blur-xl flex flex-col items-center justify-center z-50 animate-in fade-in zoom-in-95 duration-300">
                        <div className="bg-white p-4 rounded-xl mb-6 shadow-2xl relative overflow-hidden border border-slate-200">
                            <QrCode className="w-32 h-32 text-slate-900" />
                            {mobileStatus === 'CONNECTED' || mobileStatus === 'PROCESSING' || mobileStatus === 'COMPLETED' ? (
                                <div className="absolute inset-0 bg-emerald-500/90 flex items-center justify-center animate-in fade-in">
                                    <Check className="w-16 h-16 text-white" />
                                </div>
                            ) : null}
                            
                            {/* Scan Line Animation */}
                            {mobileStatus === 'WAITING' && (
                                <div className="absolute top-0 left-0 w-full h-1 bg-blue-500 shadow-[0_0_20px_#3b82f6] animate-[scan_2s_linear_infinite]"></div>
                            )}
                        </div>
                        
                        <div className="text-center space-y-2">
                            <h3 className="text-xl font-display font-bold text-slate-800">
                                {mobileStatus === 'WAITING' && 'Scan to Verify'}
                                {mobileStatus === 'CONNECTED' && 'Device Connected'}
                                {mobileStatus === 'PROCESSING' && 'Processing on Mobile...'}
                                {mobileStatus === 'COMPLETED' && 'Verification Success'}
                            </h3>
                            <p className="text-sm text-slate-500 max-w-xs mx-auto">
                                {mobileStatus === 'WAITING' ? "Camera access unavailable. Scan this QR code with your mobile device to complete verification." : "Please do not close this window."}
                            </p>
                        </div>

                   </div>
               )}

               {/* --- IDLE STATE (Light Placeholder) --- */}
               {!isCameraActive && !isMobileHandoff && (
                   <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 z-20">
                       <div className="w-16 h-16 rounded-full bg-slate-200 flex items-center justify-center mb-4 text-slate-400">
                           <PowerOff className="w-8 h-8" />
                       </div>
                       <p className="text-sm font-medium text-slate-500 mb-6">Camera Offline</p>
                       
                       <p className="text-xs text-slate-400">Initializing camera…</p>
                   </div>
               )}

               {/* --- ACTIVE OVERLAY (CAMERA MODE) --- */}
               {isCameraActive && !isMobileHandoff && (
                   <div className="absolute inset-0 pointer-events-none">
                       {/* White mask outside the guide ring */}
                       <div className="absolute inset-0 z-20">
                           <div className="absolute inset-0 bg-white/95" style={maskStyle}></div>
                       </div>
                       {/* Face Guide Ring removed */}

                       {sessionState === 'IN_PROGRESS' && multiFaceWarning && (
                         <div className="absolute top-4 left-1/2 -translate-x-1/2 z-40 pointer-events-none">
                           <div className="px-4 py-2 rounded-full bg-rose-600/90 text-white text-xs font-semibold shadow-lg">
                             Multiple faces detected. Please keep only one person in frame.
                           </div>
                       </div>
                       )}

                       {/* Instruction Text below center cutout */}
                       <div className="absolute left-1/2 top-[78%] -translate-x-1/2 z-40 flex items-center justify-center">
                           {sessionState === 'READY' && instructionText && (
                               <div className="pointer-events-none flex flex-col items-center gap-3">
                                   <p className="text-xs text-slate-700 bg-white/95 px-3 py-1.5 rounded-full border border-slate-200 shadow-sm">
                                     {instructionText}
                                   </p>
                               </div>
                           )}

                           {sessionState === 'LOADING' && (
                               <div className="flex flex-col items-center animate-in zoom-in duration-300 bg-white/95 px-4 py-2 rounded-2xl shadow-sm">
                                   <div className="mb-2 p-2 rounded-full bg-slate-100 border border-slate-200 shadow-sm">
                                     <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
                                   </div>
                                   <h2 className="text-2xl font-display font-bold text-slate-800 text-center whitespace-nowrap">
                                     {instructionText || 'Please Waiting'}
                                   </h2>
                               </div>
                           )}

                           {sessionState === 'IN_PROGRESS' && (
                               <div className="flex flex-col items-center animate-in zoom-in duration-300 bg-white/95 px-4 py-2 rounded-2xl shadow-sm">
                                   {(!isEngineReady || isLoadingModels) ? (
                                     <>
                                       <div className="mb-2 p-2 rounded-full bg-slate-100 border border-slate-200 shadow-sm">
                                         <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
                                       </div>
                                       <h2 className="text-2xl font-display font-bold text-slate-800 text-center whitespace-nowrap">
                                         {instructionText || 'Please Waiting'}
                                       </h2>
                                     </>
                                   ) : (
                                     <>
                                       <div className="mb-2 p-2 rounded-full bg-slate-100 border border-slate-200 shadow-sm">
                                           {(isStageCooldown || isChallengePassing) ? (
                                               <Check className="w-5 h-5 text-emerald-500" />
                                           ) : (
                                               React.createElement(CHALLENGES[currentChallengeIndex].icon, { className: "w-5 h-5 text-blue-500" })
                                           )}
                                       </div>
                                       <h2 className="text-2xl font-display font-bold text-slate-800 text-center whitespace-nowrap">
                                           {isStageCooldown ? "Passed" : (isChallengePassing ? "Passed" : CHALLENGES[currentChallengeIndex].instruction)}
                                       </h2>
                                     </>
                                   )}
                               </div>
                           )}
                           
                           {sessionState === 'COMPLETED' && (
                               <div className="flex flex-col items-center animate-in zoom-in duration-500 bg-white/95 px-6 py-4 rounded-2xl shadow-sm">
                                   <div className="w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center shadow-[0_0_24px_rgba(16,185,129,0.4)] mb-3">
                                       <CheckCircle2 className="w-5 h-5 text-white" />
                                   </div>
                               </div>
                           )}

                           {sessionState === 'FAILED' && (
                               <div className="flex flex-col items-center animate-in zoom-in duration-500 bg-white/95 px-6 py-4 rounded-2xl shadow-sm">
                                   <div className="w-10 h-10 rounded-full bg-rose-500 flex items-center justify-center shadow-[0_0_24px_rgba(244,63,94,0.4)] mb-3">
                                       <CheckCircle2 className="w-5 h-5 text-white" />
                                   </div>
                                   <h2 className="text-2xl font-display font-bold text-rose-500">Failed</h2>
                               </div>
                           )}
                       </div>
                   </div>
               )}

            </div>

            {isCameraActive && !isMobileHandoff && sessionState === 'IN_PROGRESS' && !isStageCooldown && !isCompletionDelay && (
              <div className="mt-3 px-2">
                <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${progressColor} transition-colors duration-300`}
                    style={{ width: `${Math.max(0, Math.min(1, stageProgress)) * 100}%` }}
                  />
                </div>
              </div>
            )}
        </div>
      </div>
    </div>
  );
};

export default FaceLiveness;
