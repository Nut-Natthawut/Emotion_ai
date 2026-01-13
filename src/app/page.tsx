"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

type CvType = any;

// Helper: แมพสีและอีโมจิตามอารมณ์ (ปรับตาม class ที่โมเดลคุณเทรนมา)
const getEmotionStyle = (emotion: string) => {
  const e = emotion.toLowerCase();
  if (e.includes("happy") || e.includes("joy")) return { color: "text-yellow-400", emoji: "😊", bg: "bg-yellow-500/20", border: "border-yellow-500/50" };
  if (e.includes("sad")) return { color: "text-blue-400", emoji: "😢", bg: "bg-blue-500/20", border: "border-blue-500/50" };
  if (e.includes("angry")) return { color: "text-red-400", emoji: "😡", bg: "bg-red-500/20", border: "border-red-500/50" };
  if (e.includes("neutral")) return { color: "text-gray-400", emoji: "😐", bg: "bg-gray-500/20", border: "border-gray-500/50" };
  if (e.includes("surprise")) return { color: "text-pink-400", emoji: "😮", bg: "bg-pink-500/20", border: "border-pink-500/50" };
  return { color: "text-white", emoji: "🤖", bg: "bg-slate-700/50", border: "border-slate-600" };
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("Initializing system...");
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);
  const [isStreaming, setIsStreaming] = useState<boolean>(false); // เพิ่ม state เช็คสถานะกล้อง

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);

  // --- Logic เดิม (Load OpenCV) ---
  async function loadOpenCV() {
    if (typeof window === "undefined") return;
    if ((window as any).cv?.Mat) {
      cvRef.current = (window as any).cv;
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/opencv/opencv.js";
      script.async = true;
      script.onload = () => {
        const cv = (window as any).cv;
        if (!cv) return reject(new Error("OpenCV โหลดแล้วแต่ window.cv ไม่มีค่า"));
        const waitReady = () => {
          if ((window as any).cv?.Mat) {
            cvRef.current = (window as any).cv;
            resolve();
          } else {
            setTimeout(waitReady, 50);
          }
        };
        if ("onRuntimeInitialized" in cv) {
          cv.onRuntimeInitialized = () => waitReady();
        } else {
          waitReady();
        }
      };
      script.onerror = () => reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));
      document.body.appendChild(script);
    });
  }

  // --- Logic เดิม (Load Cascade) ---
  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) throw new Error("cv ยังไม่พร้อม");
    const cascadeUrl = "/opencv/haarcascade_frontalface_default.xml";
    const res = await fetch(cascadeUrl);
    if (!res.ok) throw new Error("โหลด cascade ไม่สำเร็จ");
    const data = new Uint8Array(await res.arrayBuffer());
    const cascadePath = "haarcascade_frontalface_default.xml";
    try {
      cv.FS_unlink(cascadePath);
    } catch {}
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);
    const faceCascade = new cv.CascadeClassifier();
    const loaded = faceCascade.load(cascadePath);
    if (!loaded) throw new Error("cascade load() ไม่สำเร็จ");
    faceCascadeRef.current = faceCascade;
  }

  // --- Logic เดิม (Load Model) ---
  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );
    sessionRef.current = session;
    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("โหลด classes.json ไม่สำเร็จ");
    classesRef.current = await clsRes.json();
  }

  // --- Logic เดิม (Start Camera) ---
  async function startCamera() {
    setStatus("Requesting camera access...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setStatus("System Active");
      setIsStreaming(true); // เซ็ตสถานะว่ากล้องเปิดแล้ว
      requestAnimationFrame(loop);
    } catch (err) {
      setStatus("Camera access denied");
    }
  }

  // --- Logic เดิม (Preprocess) ---
  function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
    const size = 64;
    const tmp = document.createElement("canvas");
    tmp.width = size;
    tmp.height = size;
    const ctx = tmp.getContext("2d")!;
    ctx.drawImage(faceCanvas, 0, 0, size, size);
    const imgData = ctx.getImageData(0, 0, size, size).data;
    const float = new Float32Array(1 * 3 * size * size);
    let idx = 0;
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < size * size; i++) {
        const r = imgData[i * 4 + 0] / 255;
        const g = imgData[i * 4 + 1] / 255;
        const b = imgData[i * 4 + 2] / 255;
        float[idx++] = c === 0 ? r : c === 1 ? g : b;
      }
    }
    return new ort.Tensor("float32", float, [1, 3, size, size]);
  }

  // --- Logic เดิม (Softmax) ---
  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // --- Logic เดิม (Loop) ---
  async function loop() {
    try {
      const cv = cvRef.current;
      const faceCascade = faceCascadeRef.current;
      const session = sessionRef.current;
      const classes = classesRef.current;
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
        requestAnimationFrame(loop);
        return;
      }

      const ctx = canvas.getContext("2d")!;
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        requestAnimationFrame(loop);
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const faces = new cv.RectVector();
      const msize = new cv.Size(0, 0);
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

      let bestRect: any = null;
      let bestArea = 0;

      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        if (area > bestArea) {
          bestArea = area;
          bestRect = r;
        }
        // วาดกรอบให้สวยขึ้น (Cyan Glow)
        ctx.strokeStyle = "#06b6d4"; // Cyan-500
        ctx.lineWidth = 3;
        ctx.strokeRect(r.x, r.y, r.width, r.height);
      }

      if (bestRect) {
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = bestRect.width;
        faceCanvas.height = bestRect.height;
        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(canvas, bestRect.x, bestRect.y, bestRect.width, bestRect.height, 0, 0, bestRect.width, bestRect.height);

        const input = preprocessToTensor(faceCanvas);
        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = input;

        const out = await session.run(feeds);
        const outName = session.outputNames[0];
        const logits = out[outName].data as Float32Array;
        const probs = softmax(logits);
        
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        setEmotion(classes[maxIdx] ?? `class_${maxIdx}`);
        setConf(probs[maxIdx] ?? 0);

        // วาด Label ลงบน Canvas (ให้ดู Modern)
        const text = `${classes[maxIdx]} ${(probs[maxIdx] * 100).toFixed(0)}%`;
        ctx.font = "bold 18px Inter, sans-serif";
        const textMetrics = ctx.measureText(text);
        const padding = 10;
        
        // Background Label
        ctx.fillStyle = "rgba(6, 182, 212, 0.8)"; // Cyan Background
        ctx.beginPath();
        ctx.roundRect(bestRect.x, bestRect.y - 35, textMetrics.width + padding * 2, 30, 8);
        ctx.fill();

        // Text
        ctx.fillStyle = "#ffffff";
        ctx.fillText(text, bestRect.x + padding, bestRect.y - 14);
      }

      src.delete();
      gray.delete();
      faces.delete();

      requestAnimationFrame(loop);
    } catch (e: any) {
      setStatus(`Error: ${e?.message ?? e}`);
    }
  }

  // --- Boot Sequence ---
  useEffect(() => {
    (async () => {
      try {
        setStatus("Loading OpenCV...");
        await loadOpenCV();
        setStatus("Loading Haar Cascade...");
        await loadCascade();
        setStatus("Loading YOLO Model...");
        await loadModel();
        setStatus("Ready to Start");
      } catch (e: any) {
        setStatus(`Initialization Failed: ${e?.message ?? e}`);
      }
    })();
  }, []);

  const emotionStyle = getEmotionStyle(emotion);

  return (
    <main className="min-h-screen bg-[#0f172a] text-slate-100 font-sans selection:bg-cyan-500 selection:text-white overflow-hidden">
      {/* Background Glow Effects */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-cyan-600/20 rounded-full blur-[120px]" />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-10 flex flex-col items-center gap-8">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
            AI Emotion Detection
          </h1>
          <p className="text-slate-400 text-sm md:text-base">
            Real-time Facial Expression Analysis using <span className="text-cyan-400 font-medium">YOLO11</span> & <span className="text-purple-400 font-medium">ONNX Runtime</span>
          </p>
        </div>

        {/* Main Content Grid */}
        <div className="w-full grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          
          {/* Left Column: Camera Feed */}
          <div className="lg:col-span-2 w-full aspect-video bg-slate-800/50 rounded-2xl border border-slate-700/50 shadow-2xl shadow-cyan-900/20 overflow-hidden relative backdrop-blur-sm group">
            {/* Hidden Video Source */}
            <video ref={videoRef} className="hidden" playsInline muted />
            
            {/* Canvas Output */}
            <canvas
              ref={canvasRef}
              className={`w-full h-full object-cover transition-opacity duration-500 ${isStreaming ? 'opacity-100' : 'opacity-0'}`}
            />

            {/* Placeholder when not streaming */}
            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 space-y-4">
                <div className="w-16 h-16 rounded-full border-2 border-slate-600 border-dashed animate-pulse flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.818v6.364a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
                <p>Waiting for camera input...</p>
              </div>
            )}

            {/* Status Overlay */}
            <div className="absolute top-4 left-4 px-3 py-1 rounded-full bg-black/60 backdrop-blur-md text-xs font-medium border border-white/10 flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`}></span>
              {status}
            </div>
          </div>

          {/* Right Column: Dashboard & Controls */}
          <div className="w-full flex flex-col gap-6">
            
            {/* Result Card */}
            <div className={`p-6 rounded-2xl border backdrop-blur-md transition-all duration-300 ${emotionStyle.bg} ${emotionStyle.border}`}>
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold uppercase tracking-wider opacity-70">Detected Emotion</h3>
                <span className="text-3xl">{emotionStyle.emoji}</span>
              </div>
              <div className="flex items-baseline gap-2">
                <h2 className={`text-4xl font-bold ${emotionStyle.color} capitalize`}>
                  {emotion === "-" ? "Waiting..." : emotion}
                </h2>
              </div>
              
              {/* Confidence Bar */}
              <div className="mt-6 space-y-2">
                <div className="flex justify-between text-xs font-medium opacity-80">
                  <span>Confidence</span>
                  <span>{(conf * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full h-2 bg-slate-900/20 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-300 ${emotion === '-' ? 'bg-slate-500' : 'bg-current text-white'}`} 
                    style={{ width: `${conf * 100}%`, backgroundColor: emotion === '-' ? undefined : 'currentColor' }}
                  />
                </div>
              </div>
            </div>

            {/* Controls Card */}
            <div className="p-6 rounded-2xl border border-slate-700/50 bg-slate-800/30 backdrop-blur-md space-y-4">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Control Center</h3>
              
              {!isStreaming ? (
                <button
                  onClick={startCamera}
                  className="w-full py-4 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold shadow-lg shadow-cyan-500/25 transition-all active:scale-95 flex items-center justify-center gap-2"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                  </svg>
                  Start Camera Analysis
                </button>
              ) : (
                <div className="w-full py-4 rounded-xl bg-slate-700/50 border border-slate-600 text-slate-300 font-medium flex items-center justify-center gap-2 cursor-not-allowed">
                   <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  System Running...
                </div>
              )}

              <p className="text-xs text-slate-500 text-center leading-relaxed">
                By clicking start, your camera feed will be processed locally in your browser using WebAssembly. No data is sent to server.
              </p>
            </div>

          </div>
        </div>
      </div>
    </main>
  );
}