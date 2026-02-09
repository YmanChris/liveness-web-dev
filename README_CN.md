# Liveness Web（中文说明）

## 项目介绍
`liveness-web` 是一个基于浏览器的活体检测网页，通过 4 个动作完成活体验证：点头、摇头、眨眼、张嘴。流程完成后，会自动下载每个动作的 GIF 与正脸图片，并在控制台输出结构化 JSON 结果，便于业务侧抓取。

主要功能：
- 页面加载后自动开启摄像头与模型加载
- 依次执行四个动作检测
- 生成动作 GIF、输出正脸图片
- 控制台输出 JSON 结果（固定前缀方便解析）
- Vite 开发模式支持 HTTPS + 局域网访问

## 项目流程
1. **打开摄像头**（自动启动）
2. **加载模型**（ONNX 模型在浏览器中初始化）
3. **动作 1：点头**
4. **动作 2：摇头**
5. **动作 3：眨眼**
6. **动作 4：张嘴**
7. **完成/失败** → 自动下载 GIF + 正脸图片 → 控制台输出 JSON

## 全流程完成标识
- 成功：UI 出现绿色圆形图标（无文字）
- 失败：UI 出现红色圆形图标 + “Failed” 文案
- 控制台输出：前缀固定为 `[LIVENESS_RESULT]`

## 动作成功/失败标识
- **动作成功**：动作提示文案切换为 `Passed`
- **动作失败**：流程终止，进入失败状态（Failed）

## 控制台输出格式
控制台输出格式为：
```
[LIVENESS_RESULT] {JSON}
```

### JSON 字段说明
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

### 字段解释
- **event**：固定值 `LIVENESS_RESULT`，用于快速定位日志
- **status**：最终结果，`PASS` 或 `FAIL`
- **sessionState**：当前内部状态机状态
- **scores**：质量与姿态相关分数
  - `qualityScore`：综合质量分
  - `brightnessScore`：亮度评分
  - `brightnessStatus`：亮度状态（normal/dark/bright）
  - `frontalFaceScore`：正脸程度
  - `clarityScore`：清晰度评分
  - `uniformLightingScore`：光照均匀度
  - `backgroundUniformityScore`：背景均匀度
  - `pixelResolutionScore`：像素分辨率评分
- **images.faceFull**：正脸原始大图（dataURL）
- **images.faceCrop**：正脸裁剪图（dataURL）
- **images.gifs**：四个动作 GIF（dataURL）

## 下载文件命名
流程结束后，会下载以下文件：
- `nod-<timestamp>.gif`
- `shake-<timestamp>.gif`
- `blink-<timestamp>.gif`
- `open-<timestamp>.gif`
- `face-<timestamp>.png`

## 目录结构
```
liveness-web/
  src/
    components/FaceLiveness.tsx
    services/livenessEngine.ts
    models/*.onnx
    assets/onnxruntime/*
```

