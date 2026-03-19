<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { getCurrentWindow } from "@tauri-apps/api/window";
  import { onDestroy, onMount } from "svelte";

  interface TranslationPayload {
    frame_id: number;
    text: string;
    source_text: string;
    lines: TranslationLine[];
    capture_scale: number;
    ocr_backend: string;
  }

  interface TranslationLine {
    line_id: string;
    source_text: string;
    translated_text: string;
    left: number;
    top: number;
    width: number;
    height: number;
  }

  interface OcrStatusPayload {
    state: string;
    language_pair: string;
    message: string;
    backend: string;
    language_tag: string;
    used_profile_fallback: boolean;
  }

  interface OcrDiagnostics {
    backend: string;
    backend_available: boolean;
    language_pair: string;
    language_supported: boolean;
    fallback_policy: string;
  }

  interface PipelineState {
    running: boolean;
    has_task: boolean;
    task_finished: boolean;
    has_capture_region: boolean;
  }

  interface DownloadProgress {
    downloaded: number;
    total: number | null;
  }

  interface Point {
    x: number;
    y: number;
  }

  interface Rect {
    left: number;
    top: number;
    width: number;
    height: number;
  }

  type UiMode = "control" | "snip" | "capture";
  type WindowRole = "control" | "overlay";

  const LANGUAGE_STORAGE_KEY = "transit.languagePair";
  const OCR_BACKEND_STORAGE_KEY = "transit.ocrBackend";
  const languageOptions = [
    { label: "Japanese -> English", value: "ja-en" },
    { label: "Korean -> English", value: "ko-en" },
    { label: "Chinese (zh) -> English", value: "zh-en" },
  ];
  const ocrBackendOptions = [
    { label: "Auto", value: "auto" },
    { label: "Windows OCR", value: "windows" },
    { label: "Windows Profile OCR", value: "windows-profile" },
  ];

  const windowRole: WindowRole = getCurrentWindow().label === "overlay" ? "overlay" : "control";
  const expectedModelFiles = ["tokenizer.json", "encoder_model.onnx", "decoder_model_merged.onnx", "decoder_model.onnx"];

  let mode = $state<UiMode>(windowRole === "overlay" ? "capture" : "control");
  let currentLanguage = $state("ja-en");
  let modelsChecked = $state(false);
  let modelsExist = $state(false);
  let ocrSupportChecked = $state(false);
  let ocrSupported = $state(false);
  let ocrBackendSupported = $state(true);
  let selectedOcrBackend = $state("auto");
  let isDownloading = $state(false);
  let downloadProgress = $state<Record<string, DownloadProgress>>({});
  let downloadStartedAtMs = $state<number | null>(null);
  let lastSpeedSampleMs = $state<number | null>(null);
  let lastSpeedSampleBytes = $state(0);
  let downloadSpeedBytesPerSec = $state(0);

  let translatedText = $state("");
  let sourceText = $state("");
  let translatedLines = $state<TranslationLine[]>([]);
  let captureScale = $state(1);
  let ocrStatusMessage = $state("");
  let ocrStatusState = $state("idle");
  let ocrStatusLanguageTag = $state("");
  let runtimeOcrBackend = $state("auto");
  let runtimeFallbackPolicy = $state("explicit-then-profile-fallback");
  let isRunning = $state(false);
  let isClickThrough = $state(false);
  let lockedRect = $state<Rect | null>(null);
  let viewportWidth = $state(0);
  let viewportHeight = $state(0);

  let selecting = $state(false);
  let startPoint = $state<Point | null>(null);
  let selectionRect = $state<Rect | null>(null);

  let unlistenTranslation: (() => void) | null = null;
  let unlistenOcrStatus: (() => void) | null = null;
  let unlistenProgress: (() => void) | null = null;
  let unlistenOverlayMode: (() => void) | null = null;
  let unlistenClickThrough: (() => void) | null = null;
  let cleanupWindowListeners: (() => void) | null = null;
  let pipelineWatchdogTimer: ReturnType<typeof setInterval> | null = null;
  let pipelineRestartInFlight = false;
  let lastPipelineSignalAtMs = $state<number | null>(null);
  let previousFrameId = 0;
  let previousLineGeometry = $state<Record<string, TranslationLine>>({});
  let noTextStreak = $state(0);

  onMount(async () => {
    applyWindowRoleTheme();
    syncLanguageFromStorage();
    syncOcrBackendFromStorage();
    syncViewport();
    await refreshOcrBackend();
    persistOcrBackendSelection();
    await refreshModelStatus();

    mode = windowRole === "overlay" ? "capture" : "control";

    const onStorage = (event: StorageEvent) => {
      if (event.key === LANGUAGE_STORAGE_KEY && isValidLanguagePair(event.newValue)) {
        if (event.newValue !== currentLanguage) {
          currentLanguage = event.newValue;
          downloadProgress = {};
          void refreshModelStatus();
        }
      }

      if (event.key === OCR_BACKEND_STORAGE_KEY && isValidOcrBackend(event.newValue)) {
        if (event.newValue !== selectedOcrBackend) {
          selectedOcrBackend = event.newValue;
          void invoke("set_ocr_backend", { backend: selectedOcrBackend });
          void refreshModelStatus();
        }
      }
    };

    const onKeyDown = (event: KeyboardEvent) => {
      if (windowRole !== "overlay" || mode !== "snip" || event.key !== "Escape") return;
      event.preventDefault();
      void cancelSnip();
    };

    const onResize = () => {
      syncViewport();
    };

    window.addEventListener("storage", onStorage);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", onResize);
    cleanupWindowListeners = () => {
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", onResize);
      clearWindowRoleTheme();
    };

    unlistenTranslation = await listen<TranslationPayload>("translation-update", (event) => {
      if (windowRole !== "overlay") return;

      lastPipelineSignalAtMs = Date.now();

      const incomingFrameId = Number.isFinite(event.payload.frame_id) ? event.payload.frame_id : 0;
      if (incomingFrameId > 0 && incomingFrameId < previousFrameId) {
        // Backend frame IDs restart from 1 when the capture pipeline restarts.
        // Treat near-zero IDs as a new stream and reset smoothing state.
        if (incomingFrameId <= 2) {
          previousFrameId = 0;
          previousLineGeometry = {};
        } else {
          return;
        }
      }
      previousFrameId = incomingFrameId;

      translatedText = event.payload.text;
      sourceText = event.payload.source_text;
      translatedLines = smoothLines(event.payload.lines ?? []);
      captureScale = Number.isFinite(event.payload.capture_scale) && event.payload.capture_scale > 0
        ? event.payload.capture_scale
        : 1;
      runtimeOcrBackend = event.payload.ocr_backend || selectedOcrBackend;
      if (event.payload.ocr_backend === "windows-profile") {
        runtimeFallbackPolicy = "profile-only";
      }
    });

    unlistenOcrStatus = await listen<OcrStatusPayload>("ocr-status", (event) => {
      if (windowRole !== "overlay") return;

      lastPipelineSignalAtMs = Date.now();

      ocrStatusState = event.payload.state;
      ocrStatusMessage = event.payload.message;
      ocrStatusLanguageTag = event.payload.language_tag || "";
      runtimeOcrBackend = event.payload.backend || runtimeOcrBackend;
      if (event.payload.used_profile_fallback) {
        runtimeFallbackPolicy = "explicit-then-profile-fallback";
      }

      if (event.payload.state === "no-text") {
        noTextStreak += 1;
        const lang = event.payload.language_pair;
        if (lang === "ja-en" || lang === "zh-en" || lang === "ko-en") {
          ocrStatusMessage = `${event.payload.message}. If this is ${lang.split("-")[0]}, ensure the Windows OCR language is installed.`;
        }

        // Prevent stale overlays from lingering when OCR repeatedly sees no text.
        if (noTextStreak >= 3) {
          translatedLines = [];
          translatedText = "";
          sourceText = "";
          previousLineGeometry = {};
        }
      } else if (event.payload.state === "ok") {
        noTextStreak = 0;
      } else if (event.payload.state === "warning") {
        noTextStreak = 0;
        if (event.payload.used_profile_fallback) {
          const suffix = event.payload.language_tag ? ` (effective OCR: ${event.payload.language_tag})` : "";
          ocrStatusMessage = `${event.payload.message}${suffix}`;
        }
      } else if (event.payload.state === "ignored") {
        noTextStreak = 0;
        translatedLines = [];
        translatedText = "";
        sourceText = "";
        previousLineGeometry = {};
        ocrStatusMessage = "";
      } else if (event.payload.state === "error") {
        noTextStreak = Math.min(noTextStreak + 1, 99);
      }
    });

    unlistenProgress = await listen<{
      file: string;
      downloaded: number;
      total: number | null;
    }>("download-progress", (event) => {
      if (windowRole !== "control") return;
      const p = event.payload;
      downloadProgress[p.file] = { downloaded: p.downloaded, total: p.total };
      updateDownloadSpeedEstimate();
    });

    unlistenOverlayMode = await listen<string>("overlay-mode", (event) => {
      if (windowRole !== "overlay") return;
      if (event.payload === "snip") {
        mode = "snip";
        selecting = false;
        startPoint = null;
        selectionRect = null;
        lockedRect = null;
        translatedText = "";
        sourceText = "";
        translatedLines = [];
        previousFrameId = 0;
        previousLineGeometry = {};
        noTextStreak = 0;
        ocrStatusLanguageTag = "";
        lastPipelineSignalAtMs = null;
      }
    });

    unlistenClickThrough = await listen<boolean>("click-through-changed", (event) => {
      isClickThrough = !!event.payload;
    });

    if (windowRole === "overlay") {
      pipelineWatchdogTimer = setInterval(() => {
        if (mode !== "capture" || !isRunning || pipelineRestartInFlight) {
          return;
        }

        void probePipelineStateAndRecover();
      }, 1500);
    }
  });

  onDestroy(() => {
    if (unlistenTranslation) unlistenTranslation();
    if (unlistenOcrStatus) unlistenOcrStatus();
    if (unlistenProgress) unlistenProgress();
    if (unlistenOverlayMode) unlistenOverlayMode();
    if (unlistenClickThrough) unlistenClickThrough();
    if (pipelineWatchdogTimer) clearInterval(pipelineWatchdogTimer);
    if (cleanupWindowListeners) cleanupWindowListeners();
  });

  async function restartPipelineFromWatchdog() {
    if (pipelineRestartInFlight || windowRole !== "overlay") return;

    pipelineRestartInFlight = true;
    ocrStatusState = "warning";
    ocrStatusMessage = "Capture pipeline stalled, restarting...";

    try {
      await invoke("stop_translation");
      previousFrameId = 0;
      previousLineGeometry = {};
      await invoke("start_translation", { languagePair: currentLanguage, ocrBackend: selectedOcrBackend });
      lastPipelineSignalAtMs = Date.now();
      isRunning = true;
    } catch (e) {
      ocrStatusState = "error";
      ocrStatusMessage = "Failed to restart capture pipeline";
      console.error("Watchdog restart failed:", e);
    } finally {
      pipelineRestartInFlight = false;
    }
  }

  async function probePipelineStateAndRecover() {
    if (pipelineRestartInFlight || windowRole !== "overlay" || mode !== "capture" || !isRunning) {
      return;
    }

    try {
      const state = await invoke<PipelineState>("get_pipeline_state");
      if (!state.has_capture_region) {
        ocrStatusState = "warning";
        ocrStatusMessage = "Capture region missing, please re-capture";
        return;
      }

      if (!state.running || !state.has_task || state.task_finished) {
        await restartPipelineFromWatchdog();
      }
    } catch (e) {
      console.error("Pipeline state probe failed:", e);
    }
  }

  function applyWindowRoleTheme() {
    document.documentElement.dataset.windowRole = windowRole;
    document.body.dataset.windowRole = windowRole;
  }

  function syncViewport() {
    viewportWidth = window.innerWidth;
    viewportHeight = window.innerHeight;
  }

  function clearWindowRoleTheme() {
    delete document.documentElement.dataset.windowRole;
    delete document.body.dataset.windowRole;
  }

  function isValidLanguagePair(value: string | null): value is string {
    return !!value && languageOptions.some((option) => option.value === value);
  }

  function syncLanguageFromStorage() {
    try {
      const stored = localStorage.getItem(LANGUAGE_STORAGE_KEY);
      if (isValidLanguagePair(stored)) {
        currentLanguage = stored;
      } else {
        persistLanguageSelection();
      }
    } catch (e) {
      console.warn("Language storage unavailable:", e);
    }
  }

  function persistLanguageSelection() {
    try {
      localStorage.setItem(LANGUAGE_STORAGE_KEY, currentLanguage);
    } catch (e) {
      console.warn("Language storage unavailable:", e);
    }
  }

  function isValidOcrBackend(value: string | null): value is string {
    return !!value && ocrBackendOptions.some((option) => option.value === value);
  }

  function syncOcrBackendFromStorage() {
    try {
      const stored = localStorage.getItem(OCR_BACKEND_STORAGE_KEY);
      if (isValidOcrBackend(stored)) {
        selectedOcrBackend = stored;
      } else {
        persistOcrBackendSelection();
      }
    } catch (e) {
      console.warn("OCR backend storage unavailable:", e);
    }
  }

  function persistOcrBackendSelection() {
    try {
      localStorage.setItem(OCR_BACKEND_STORAGE_KEY, selectedOcrBackend);
    } catch (e) {
      console.warn("OCR backend storage unavailable:", e);
    }
  }

  async function refreshOcrBackend() {
    try {
      const backend = await invoke<string>("get_ocr_backend");
      selectedOcrBackend = backend || "auto";
    } catch {
      selectedOcrBackend = "auto";
    }
  }

  async function refreshModelStatus() {
    modelsChecked = false;
    ocrSupportChecked = false;
    try {
      modelsExist = await invoke<boolean>("check_models", { languagePair: currentLanguage });
      const diagnostics = await invoke<OcrDiagnostics>("get_ocr_diagnostics", {
        languagePair: currentLanguage,
        backend: selectedOcrBackend,
      });

      selectedOcrBackend = diagnostics.backend || selectedOcrBackend;
      ocrSupported = diagnostics.language_supported;
      ocrBackendSupported = diagnostics.backend_available;
      runtimeFallbackPolicy = diagnostics.fallback_policy || runtimeFallbackPolicy;
    } catch (e) {
      console.error("Model check failed:", e);
      modelsExist = false;
      ocrSupported = false;
      ocrBackendSupported = false;
    } finally {
      modelsChecked = true;
      ocrSupportChecked = true;
    }
  }

  async function onOcrBackendChange(event: Event) {
    const select = event.currentTarget as HTMLSelectElement;
    selectedOcrBackend = select.value;
    persistOcrBackendSelection();
    try {
      await invoke("set_ocr_backend", { backend: selectedOcrBackend });
    } catch (e) {
      console.error("Failed to set OCR backend:", e);
      alert("Failed to set OCR backend: " + e);
    }
    await refreshModelStatus();
  }

  async function onLanguageChange(event: Event) {
    const select = event.currentTarget as HTMLSelectElement;
    currentLanguage = select.value;
    persistLanguageSelection();
    downloadProgress = {};
    await refreshModelStatus();
  }

  async function installSelectedLanguage() {
    await installLanguage(false);
  }

  async function reinstallSelectedLanguage() {
    await installLanguage(true);
  }

  async function installLanguage(force: boolean) {
    persistLanguageSelection();
    isDownloading = true;
    downloadProgress = {};
    downloadStartedAtMs = Date.now();
    lastSpeedSampleMs = downloadStartedAtMs;
    lastSpeedSampleBytes = 0;
    downloadSpeedBytesPerSec = 0;
    try {
      await invoke("download_models", { languagePair: currentLanguage });
      await invoke("init_engine", { languagePair: currentLanguage });
      modelsExist = true;
    } catch (e) {
      console.error("Failed to install model:", e);
      alert("Failed to " + (force ? "reinstall" : "install") + " model: " + e);
    } finally {
      await refreshModelStatus();
      isDownloading = false;
      downloadSpeedBytesPerSec = 0;
      lastSpeedSampleMs = null;
      lastSpeedSampleBytes = 0;
    }
  }

  async function startCapture() {
    if (windowRole !== "control" && windowRole !== "overlay") return;

    syncLanguageFromStorage();
    await refreshModelStatus();

    if (!modelsExist || isDownloading) {
      alert("Install the selected language model first.");
      return;
    }

    if (!ocrSupported) {
      alert("Windows OCR language support is missing for the selected language. Install the matching Windows language OCR pack first.");
      return;
    }

    if (!ocrBackendSupported) {
      alert("The selected OCR backend is not available on this system.");
      return;
    }

    try {
      await invoke("init_engine", { languagePair: currentLanguage });
      if (isRunning) {
        await invoke("stop_translation");
        isRunning = false;
      }
      if (isClickThrough) {
        isClickThrough = false;
        await invoke("set_click_through", { enabled: false });
      }

      translatedText = "";
      sourceText = "";
      translatedLines = [];
      captureScale = 1;
      ocrStatusMessage = "";
      ocrStatusState = "idle";
      previousFrameId = 0;
      previousLineGeometry = {};
      noTextStreak = 0;
      ocrStatusLanguageTag = "";
      lockedRect = null;
      selecting = false;
      startPoint = null;
      selectionRect = null;

      await invoke("begin_new_capture");
    } catch (e) {
      console.error("Failed to start capture:", e);
      alert("Failed to start capture: " + e);
    }
  }

  function getRectFromPoints(a: Point, b: Point): Rect {
    const left = Math.min(a.x, b.x);
    const top = Math.min(a.y, b.y);
    const width = Math.abs(a.x - b.x);
    const height = Math.abs(a.y - b.y);
    return { left, top, width, height };
  }

  function onSnipPointerDown(event: PointerEvent) {
    if (mode !== "snip") return;

    const target = event.target as HTMLElement | null;
    if (target?.closest("button")) return;

    const point = { x: event.clientX, y: event.clientY };
    startPoint = point;
    selectionRect = { left: point.x, top: point.y, width: 0, height: 0 };
    selecting = true;
  }

  function onSnipPointerMove(event: PointerEvent) {
    if (mode !== "snip" || !selecting || !startPoint) return;
    selectionRect = getRectFromPoints(startPoint, { x: event.clientX, y: event.clientY });
  }

  async function onSnipPointerUp(event: PointerEvent) {
    if (mode !== "snip" || !selecting || !startPoint) return;
    selecting = false;

    const rect = getRectFromPoints(startPoint, { x: event.clientX, y: event.clientY });
    if (rect.width < 8 || rect.height < 8) {
      startPoint = null;
      selectionRect = null;
      return;
    }

    try {
      await invoke("finish_new_capture", {
        x: Math.round(rect.left),
        y: Math.round(rect.top),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        scaleFactor: window.devicePixelRatio || 1,
      });
      await invoke("start_translation", { languagePair: currentLanguage, ocrBackend: selectedOcrBackend });

      ocrStatusState = "warning";
      ocrStatusMessage = "Initializing capture pipeline...";
      lastPipelineSignalAtMs = Date.now();
      isRunning = true;
      lockedRect = rect;
      mode = "capture";
      startPoint = null;
      selectionRect = null;
    } catch (e) {
      console.error("Failed to finalize capture:", e);
      alert("Failed to finalize capture: " + e);
    }
  }

  async function cancelSnip() {
    try {
      await invoke("cancel_new_capture");
    } catch (e) {
      console.error("Failed to cancel capture:", e);
      alert("Failed to cancel capture: " + e);
    } finally {
      mode = "capture";
      selecting = false;
      startPoint = null;
      selectionRect = null;
      lockedRect = null;
    }
  }

  async function toggleCapturePause() {
    if (windowRole !== "overlay") return;
    if (isRunning) {
      await invoke("stop_translation");
      isRunning = false;
    } else {
      await invoke("start_translation", { languagePair: currentLanguage, ocrBackend: selectedOcrBackend });
      ocrStatusState = "warning";
      ocrStatusMessage = "Initializing capture pipeline...";
      lastPipelineSignalAtMs = Date.now();
      isRunning = true;
    }
  }

  async function setClickThroughEnabled(enabled: boolean) {
    const previous = isClickThrough;
    isClickThrough = enabled;

    try {
      await invoke("set_click_through", { enabled });
    } catch (e) {
      isClickThrough = previous;
      console.error("Failed to set click-through:", e);
      alert("Failed to set click-through: " + e);
    }
  }

  async function toggleClickThrough() {
    if (windowRole !== "overlay") return;
    await setClickThroughEnabled(!isClickThrough);
  }

  async function disableClickThroughFromControl() {
    await setClickThroughEnabled(false);
  }

  async function unlockCaptureFromControl() {
    if (windowRole !== "control") return;

    try {
      await invoke("stop_translation");
      if (isClickThrough) {
        await setClickThroughEnabled(false);
      }
      await invoke("unlock_capture");
      isRunning = false;
      lockedRect = null;
    } catch (e) {
      console.error("Failed to unlock capture from control:", e);
      alert("Failed to unlock capture: " + e);
    }
  }

  async function unlockCapture() {
    if (windowRole !== "overlay") return;

    if (isRunning) {
      await invoke("stop_translation");
      isRunning = false;
    }

    await invoke("unlock_capture");
    if (isClickThrough) {
      await setClickThroughEnabled(false);
    }

    translatedText = "";
    sourceText = "";
    translatedLines = [];
    captureScale = 1;
    ocrStatusMessage = "";
    ocrStatusState = "idle";
    previousFrameId = 0;
    previousLineGeometry = {};
    noTextStreak = 0;
    ocrStatusLanguageTag = "";
    lastPipelineSignalAtMs = null;
    mode = "capture";
    selecting = false;
    startPoint = null;
    selectionRect = null;
    lockedRect = null;
  }

  function getDownloadTotals(): { downloaded: number; total: number } {
    const entries = Object.entries(downloadProgress)
      .filter(([file]) => expectedModelFiles.includes(file))
      .map(([, p]) => p)
      .filter((p) => typeof p.total === "number") as Array<{ downloaded: number; total: number }>;

    const downloaded = entries.reduce((sum, p) => sum + p.downloaded, 0);
    const total = entries.reduce((sum, p) => sum + p.total, 0);
    return { downloaded, total };
  }

  function getDownloadPercent(): number {
    const { downloaded, total } = getDownloadTotals();
    if (total <= 0) {
      if (!isDownloading) return 0;

      const hasAnyDownloadedBytes = Object.values(downloadProgress).some((p) => p.downloaded > 0);
      if (!hasAnyDownloadedBytes) return 0;

      const elapsedSec = downloadStartedAtMs ? (Date.now() - downloadStartedAtMs) / 1000 : 0;
      return Math.min(95, 5 + elapsedSec * 1.8);
    }
    return Math.min(100, (downloaded / total) * 100);
  }

  function updateDownloadSpeedEstimate() {
    const now = Date.now();
    const { downloaded, total } = getDownloadTotals();

    if (lastSpeedSampleMs === null) {
      lastSpeedSampleMs = now;
      lastSpeedSampleBytes = downloaded;
      return;
    }

    const elapsedMs = now - lastSpeedSampleMs;
    if (elapsedMs < 400) return;

    const bytesDelta = Math.max(0, downloaded - lastSpeedSampleBytes);
    const currentBps = elapsedMs > 0 ? (bytesDelta * 1000) / elapsedMs : 0;

    // Smooth out spikes with EMA so ETA is less jumpy.
    downloadSpeedBytesPerSec = downloadSpeedBytesPerSec > 0
      ? downloadSpeedBytesPerSec * 0.7 + currentBps * 0.3
      : currentBps;

    lastSpeedSampleMs = now;
    lastSpeedSampleBytes = downloaded;

    if (total > 0 && downloaded >= total) {
      downloadSpeedBytesPerSec = 0;
    }
  }

  function formatSpeed(bytesPerSec: number): string {
    if (!Number.isFinite(bytesPerSec) || bytesPerSec <= 0) return "Calculating speed...";
    const mbps = bytesPerSec / (1024 * 1024);
    return `${mbps.toFixed(2)} MB/s`;
  }

  function formatEta(): string {
    const { downloaded, total } = getDownloadTotals();
    if (total <= 0 || downloadSpeedBytesPerSec <= 0 || downloaded >= total) return "ETA --";

    const remainingSec = Math.max(0, (total - downloaded) / downloadSpeedBytesPerSec);
    const mins = Math.floor(remainingSec / 60);
    const secs = Math.floor(remainingSec % 60);
    if (mins > 0) return `ETA ${mins}m ${secs}s`;
    return `ETA ${secs}s`;
  }

  function formatElapsed(): string {
    if (!downloadStartedAtMs) return "Elapsed --";
    const elapsedSec = Math.max(0, Math.floor((Date.now() - downloadStartedAtMs) / 1000));
    const mins = Math.floor(elapsedSec / 60);
    const secs = elapsedSec % 60;
    if (mins > 0) return `Elapsed ${mins}m ${secs}s`;
    return `Elapsed ${secs}s`;
  }

  function getSelectionStyle(rect: Rect): string {
    return `left: ${rect.left}px; top: ${rect.top}px; width: ${rect.width}px; height: ${rect.height}px;`;
  }

  function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }

  function getCaptureFrameStyle(rect: Rect | null): string {
    if (!rect) return "display: none;";
    return getSelectionStyle(rect);
  }

  function getCaptureReadoutStyle(rect: Rect | null): string {
    if (!rect) {
      return "left: 12px; right: 12px; bottom: 12px;";
    }

    const maxWidth = Math.max(220, viewportWidth - 24);
    const width = Math.min(Math.max(rect.width, 260), Math.min(560, maxWidth));
    const left = clamp(rect.left, 12, Math.max(12, viewportWidth - width - 12));
    const estimatedHeight = sourceText ? 170 : translatedText ? 118 : 72;
    const spaceBelow = viewportHeight - (rect.top + rect.height) - 12;
    const spaceAbove = rect.top - 12;

    let top = rect.top + rect.height + 12;
    if (spaceBelow < estimatedHeight && spaceAbove >= estimatedHeight) {
      top = rect.top - estimatedHeight - 12;
    } else if (spaceBelow < estimatedHeight && spaceAbove < estimatedHeight) {
      top = clamp(viewportHeight - estimatedHeight - 12, 12, Math.max(12, viewportHeight - estimatedHeight - 12));
    }

    return `left: ${left}px; top: ${top}px; width: ${width}px;`;
  }

  function getInPlaceLayerStyle(rect: Rect | null): string {
    if (!rect) return "display: none;";
    return getSelectionStyle(rect);
  }

  function getLineOverlayStyle(line: TranslationLine): string {
    const dpr = captureScale > 0 ? captureScale : window.devicePixelRatio || 1;
    const left = Math.max(0, line.left / dpr);
    const top = Math.max(0, line.top / dpr);
    const width = Math.max(8, line.width / dpr);
    const height = Math.max(16, line.height / dpr);

    const text = (line.translated_text || "").trim();
    const glyphCount = Math.max(1, Array.from(text).length);
    const looksCjk = /[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/.test(text);

    const computeFont = (w: number, h: number): number => {
      const area = Math.max(1, w * h);
      const areaBased = Math.sqrt(area / (glyphCount * (looksCjk ? 1.3 : 1.8)));
      const heightCap = h * (looksCjk ? 0.78 : 0.72);
      return Math.max(11, Math.min(38, Math.min(areaBased, heightCap)));
    };

    if (lockedRect) {
      const maxW = Math.max(8, lockedRect.width - left);
      const maxH = Math.max(16, lockedRect.height - top);
      const boundedW = Math.max(8, Math.min(width, maxW));
      const boundedH = Math.max(16, Math.min(height, maxH));
      const fontSize = computeFont(boundedW, boundedH);
      const padY = Math.max(1, Math.round(boundedH * 0.08));
      const padX = Math.max(2, Math.round(boundedH * 0.15));
      const letterSpacing = looksCjk ? 0 : 0.2;
      return `left:${left}px;top:${top}px;width:${boundedW}px;height:${boundedH}px;font-size:${fontSize}px;padding:${padY}px ${padX}px;letter-spacing:${letterSpacing}px;`;
    }

    const fontSize = computeFont(width, height);
    const padY = Math.max(1, Math.round(height * 0.08));
    const padX = Math.max(2, Math.round(height * 0.15));
    const letterSpacing = looksCjk ? 0 : 0.2;
    return `left:${left}px;top:${top}px;width:${width}px;height:${height}px;font-size:${fontSize}px;padding:${padY}px ${padX}px;letter-spacing:${letterSpacing}px;`;
  }

  function computeIou(a: TranslationLine, b: TranslationLine): number {
    const ax1 = a.left;
    const ay1 = a.top;
    const ax2 = a.left + a.width;
    const ay2 = a.top + a.height;

    const bx1 = b.left;
    const by1 = b.top;
    const bx2 = b.left + b.width;
    const by2 = b.top + b.height;

    const ix1 = Math.max(ax1, bx1);
    const iy1 = Math.max(ay1, by1);
    const ix2 = Math.min(ax2, bx2);
    const iy2 = Math.min(ay2, by2);

    if (ix2 <= ix1 || iy2 <= iy1) return 0;

    const inter = (ix2 - ix1) * (iy2 - iy1);
    const aArea = Math.max(1, (ax2 - ax1) * (ay2 - ay1));
    const bArea = Math.max(1, (bx2 - bx1) * (by2 - by1));
    const union = aArea + bArea - inter;

    return union > 0 ? inter / union : 0;
  }

  function smoothLines(lines: TranslationLine[]): TranslationLine[] {
    const alpha = 0.5;
    const nextGeometry: Record<string, TranslationLine> = {};
    const consumedPrevIds: Record<string, boolean> = {};

    const smoothed = lines.map((line) => {
      const defaultId = line.line_id || `${line.left}:${line.top}:${line.source_text}`;

      let stableId = defaultId;
      let prev = previousLineGeometry[stableId];

      if (!prev) {
        let bestId = "";
        let bestScore = 0;
        for (const [candidateId, candidate] of Object.entries(previousLineGeometry)) {
          if (consumedPrevIds[candidateId]) continue;
          const iou = computeIou(line, candidate);
          if (iou > bestScore) {
            bestScore = iou;
            bestId = candidateId;
          }
        }

        if (bestId && bestScore >= 0.35) {
          stableId = bestId;
          prev = previousLineGeometry[bestId];
        }
      }

      const merged: TranslationLine = prev
        ? {
            ...line,
            line_id: stableId,
            left: Math.round(prev.left * (1 - alpha) + line.left * alpha),
            top: Math.round(prev.top * (1 - alpha) + line.top * alpha),
            width: Math.round(prev.width * (1 - alpha) + line.width * alpha),
            height: Math.round(prev.height * (1 - alpha) + line.height * alpha),
          }
        : { ...line, line_id: stableId };

      nextGeometry[stableId] = merged;
      consumedPrevIds[stableId] = true;
      return merged;
    });

    previousLineGeometry = nextGeometry;
    return smoothed;
  }

  function formatSelectionSize(rect: Rect | null): string {
    if (!rect) return "";
    return `${Math.round(rect.width)} x ${Math.round(rect.height)}`;
  }

  function stopEvent(event: Event) {
    event.stopPropagation();
  }
</script>

<main
  class="surface"
  class:control-surface={windowRole === "control"}
  class:overlay-surface={windowRole === "overlay"}
  role="application"
>
  {#if windowRole === "control"}
    <section class="control-module" data-tauri-drag-region>
      <header class="module-head" data-tauri-drag-region>
        <h1>Transit Control</h1>
        <span class="badge">Ready-then-Capture</span>
      </header>

      <div class="module-body">
        <label for="lang">Translation Language</label>
        <select id="lang" value={currentLanguage} onchange={onLanguageChange}>
          {#each languageOptions as option (option.value)}
            <option value={option.value}>{option.label}</option>
          {/each}
        </select>

        <label for="ocr-backend">OCR Backend</label>
        <select id="ocr-backend" value={selectedOcrBackend} onchange={onOcrBackendChange}>
          {#each ocrBackendOptions as option (option.value)}
            <option value={option.value}>{option.label}</option>
          {/each}
        </select>

        <div class="status-row">
          <span class="dot" class:ok={modelsExist}></span>
          <span>{modelsChecked ? (modelsExist ? "Model installed" : "Model not installed") : "Checking model..."}</span>
        </div>

        <div class="status-row">
          <span class="dot" class:ok={ocrSupported}></span>
          <span>
            {ocrSupportChecked
              ? (ocrSupported ? "OCR language ready" : "OCR language missing in Windows")
              : "Checking OCR support..."}
          </span>
        </div>

        <div class="status-row">
          <span class="dot" class:ok={ocrBackendSupported}></span>
          <span>{ocrBackendSupported ? `OCR backend ready (${selectedOcrBackend})` : `OCR backend unavailable (${selectedOcrBackend})`}</span>
        </div>

        {#if ocrSupportChecked && !ocrSupported}
          <div class="ocr-warning">
            Install Windows OCR support for {currentLanguage.split("-")[0]} before capture.
          </div>
        {/if}

        <div class="actions">
          <button onclick={installSelectedLanguage} disabled={isDownloading}>
            {isDownloading ? "Installing..." : modelsExist ? "Installed" : "Install Model"}
          </button>
          <button onclick={reinstallSelectedLanguage} disabled={isDownloading}>
            Reinstall
          </button>
          <button class="primary" onclick={startCapture} disabled={!modelsExist || isDownloading || !ocrSupported || !ocrBackendSupported}>
            Capture
          </button>
        </div>

        <div class="actions">
          <button onclick={disableClickThroughFromControl} disabled={!isClickThrough}>
            Disable Click-through
          </button>
          <button onclick={unlockCaptureFromControl}>
            Unlock Capture
          </button>
        </div>

        {#if isDownloading}
          <div class="progress-list">
            <div class="overall-progress">
              <div class="progress-meta">
                <span>Downloading model files...</span>
                <span>{getDownloadPercent().toFixed(1)}%</span>
              </div>
              <div class="bar overall"><div class="fill" style="width: {getDownloadPercent()}%"></div></div>
              <div class="download-metrics">
                <span>{formatSpeed(downloadSpeedBytesPerSec)}</span>
                <span>{formatEta()}</span>
                <span>{formatElapsed()}</span>
              </div>
            </div>

            {#if Object.keys(downloadProgress).length === 0}
              <div class="progress-empty">Preparing download stream...</div>
            {/if}

            {#each Object.entries(downloadProgress) as [file, p] (file)}
              <div class="progress-item">
                <div class="progress-meta">
                  <span>{file}</span>
                  <span>
                    {(p.downloaded / 1024 / 1024).toFixed(1)}MB
                    {#if p.total} / {(p.total / 1024 / 1024).toFixed(1)}MB{/if}
                  </span>
                </div>
                {#if p.total}
                  <div class="bar"><div class="fill" style="width: {(p.downloaded / p.total) * 100}%"></div></div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </section>
  {/if}

  {#if windowRole === "overlay" && mode === "capture"}
    <section class="capture-layer">
      {#if lockedRect}
        <div class="capture-frame" style={getCaptureFrameStyle(lockedRect)}></div>
      {/if}

      <div class="capture-toolbar">
        <span class="capture-status" class:paused={!isRunning}>{isRunning ? "Live" : "Paused"}</span>
        <span class="capture-backend">OCR: {runtimeOcrBackend}</span>
        <span class="capture-backend policy">{runtimeFallbackPolicy}</span>
        <button onclick={toggleCapturePause}>{isRunning ? "Pause" : "Resume"}</button>
        <button onclick={toggleClickThrough}>{isClickThrough ? "Unlock UI" : "Click-through"}</button>
        <button onclick={startCapture}>Re-capture</button>
        <button class="danger" onclick={unlockCapture}>Unlock</button>
      </div>

      {#if lockedRect}
        <div class="in-place-layer" style={getInPlaceLayerStyle(lockedRect)}>
          {#each translatedLines as line, idx (`${line.line_id}:${idx}`)}
            {#if line.translated_text?.trim()}
              <div class="in-place-line" style={getLineOverlayStyle(line)}>{line.translated_text}</div>
            {/if}
          {/each}
        </div>
      {/if}

      {#if !translatedLines.length && ocrStatusState !== "ignored"}
        <div class="capture-readout" style={getCaptureReadoutStyle(lockedRect)}>
          <div class="capture-placeholder">
            {#if isRunning}
              <span class:warn={ocrStatusState === "warning"} class:error={ocrStatusState === "error"}>
                {ocrStatusState === "error" ? ocrStatusMessage : (ocrStatusMessage || "Watching selected region for text...")}
              </span>
            {:else}
              Translation paused.
            {/if}
          </div>
        </div>
      {/if}
    </section>
  {/if}

  {#if windowRole === "overlay" && mode === "snip"}
    <div
      class="snip-overlay"
      role="application"
      onpointerdown={onSnipPointerDown}
      onpointermove={onSnipPointerMove}
      onpointerup={onSnipPointerUp}
    >
      <div class="snip-hint">
        <span>Drag a live translation area</span>
        {#if selectionRect}
          <span class="snip-size">{formatSelectionSize(selectionRect)}</span>
        {/if}
        <button
          onpointerdown={stopEvent}
          onclick={async (event) => {
            stopEvent(event);
            await cancelSnip();
          }}
        >
          Cancel
        </button>
      </div>

      {#if selectionRect}
        <div class="snip-rect" style={getSelectionStyle(selectionRect)}>
          <div class="snip-size-badge">{formatSelectionSize(selectionRect)}</div>
        </div>
      {/if}
    </div>
  {/if}
</main>

<style>
  :global(*) {
    box-sizing: border-box;
  }

  :global(html),
  :global(body) {
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #0d1119;
    font-family: "Segoe UI", system-ui, sans-serif;
  }

  :global(html[data-window-role="overlay"]),
  :global(body[data-window-role="overlay"]) {
    background: transparent;
  }

  .surface {
    position: fixed;
    inset: 0;
    user-select: none;
  }

  .control-surface {
    background:
      radial-gradient(circle at 15% 15%, rgba(51, 132, 255, 0.16), transparent 42%),
      radial-gradient(circle at 80% 20%, rgba(25, 182, 255, 0.1), transparent 34%),
      linear-gradient(165deg, #0d1119, #0b1220);
  }

  .overlay-surface {
    background: transparent;
  }

  .control-module {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    margin: 0;
    border-radius: 0;
    background: rgba(12, 14, 20, 0.93);
    border: none;
    box-shadow: none;
    color: #ecf1f8;
  }

  .module-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.09);
    -webkit-app-region: drag;
  }

  .module-head h1 {
    font-size: 15px;
    margin: 0;
    font-weight: 700;
  }

  .badge {
    font-size: 11px;
    color: #7bd3ff;
  }

  .module-body {
    padding: 14px 16px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    height: calc(100% - 54px);
    overflow: auto;
  }

  label {
    font-size: 12px;
    color: #b8c2cf;
  }

  select,
  button {
    -webkit-app-region: no-drag;
  }

  select {
    width: 100%;
    height: 36px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(27, 32, 43, 0.9);
    color: #ecf1f8;
    padding: 0 10px;
  }

  .status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #b8c2cf;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: #f59e0b;
  }

  .dot.ok {
    background: #34d399;
  }

  .actions {
    display: flex;
    gap: 8px;
  }

  button {
    border: none;
    border-radius: 10px;
    height: 36px;
    padding: 0 12px;
    background: rgba(255, 255, 255, 0.09);
    color: #ecf1f8;
    cursor: pointer;
    transition: 0.15s ease;
  }

  button:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.18);
  }

  button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  button.primary {
    background: #19b6ff;
    color: #021623;
    font-weight: 700;
  }

  button.primary:hover:not(:disabled) {
    background: #4fc8ff;
  }

  .progress-list {
    margin-top: 4px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 10px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .progress-empty {
    font-size: 12px;
    color: #9fb0c5;
  }

  .overall-progress {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .download-metrics {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 11px;
    color: #a8bed7;
    margin-top: 4px;
  }

  .progress-meta {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #b8c2cf;
  }

  .bar {
    margin-top: 4px;
    height: 4px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.14);
  }

  .bar.overall {
    height: 7px;
    background: rgba(255, 255, 255, 0.18);
  }

  .fill {
    height: 100%;
    background: #21b8ff;
    transition: width 0.08s linear;
  }

  .capture-layer {
    position: fixed;
    inset: 0;
    background: transparent;
  }

  .capture-frame {
    position: fixed;
    border-radius: 18px;
    border: 2px solid rgba(117, 216, 255, 0.9);
    box-shadow:
      0 0 0 1px rgba(255, 255, 255, 0.25) inset,
      0 18px 40px rgba(0, 0, 0, 0.22);
    pointer-events: none;
  }

  .in-place-layer {
    position: fixed;
    border-radius: 14px;
    pointer-events: none;
    overflow: hidden;
  }

  .in-place-line {
    position: absolute;
    display: block;
    line-height: 1.16;
    font-weight: 700;
    color: #111822;
    background: rgba(245, 249, 255, 0.94);
    border-radius: 6px;
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.48), 0 0 0 1px rgba(120, 138, 168, 0.22);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
    text-overflow: ellipsis;
    overflow: hidden;
    transition:
      left 120ms linear,
      top 120ms linear,
      width 120ms linear,
      height 120ms linear,
      font-size 120ms linear;
  }

  .ocr-warning {
    border-radius: 10px;
    padding: 8px 10px;
    border: 1px solid rgba(255, 181, 111, 0.45);
    background: rgba(255, 181, 111, 0.12);
    color: #ffd8a8;
    font-size: 11px;
    line-height: 1.35;
  }

  .capture-toolbar {
    position: fixed;
    top: 10px;
    right: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px;
    border-radius: 999px;
    background: rgba(7, 12, 22, 0.64);
    border: 1px solid rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(10px);
  }

  .capture-status {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 54px;
    height: 28px;
    padding: 0 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #052034;
    background: #7be0ff;
  }

  .capture-status.paused {
    color: #ffe8b8;
    background: rgba(245, 158, 11, 0.28);
  }

  .capture-backend {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 28px;
    padding: 0 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    color: #cfe9ff;
    background: rgba(49, 84, 130, 0.34);
  }

  .capture-backend.policy {
    color: #d8f6dc;
    background: rgba(42, 108, 62, 0.32);
  }

  .capture-toolbar button {
    height: 28px;
    border-radius: 999px;
    padding: 0 11px;
    font-size: 12px;
    background: rgba(255, 255, 255, 0.11);
  }

  .capture-toolbar .danger {
    background: rgba(255, 82, 82, 0.25);
    color: #ffd8d8;
  }

  .capture-readout {
    position: fixed;
    display: flex;
    flex-direction: column;
    gap: 8px;
    pointer-events: none;
    z-index: 1;
  }

  .capture-placeholder {
    border-radius: 18px;
    padding: 12px 14px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(5, 10, 19, 0.6);
    backdrop-filter: blur(12px);
    color: #ecf6ff;
  }

  .capture-placeholder {
    font-size: 12px;
    color: rgba(236, 246, 255, 0.86);
  }

  .capture-placeholder .warn {
    color: #ffd9a6;
  }

  .capture-placeholder .error {
    color: #ffc7c7;
  }

  .snip-overlay {
    position: fixed;
    inset: 0;
    background: rgba(5, 10, 18, 0.05);
    cursor: crosshair;
  }

  .snip-overlay::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(123, 224, 255, 0.06) 1px, transparent 1px),
      linear-gradient(90deg, rgba(123, 224, 255, 0.06) 1px, transparent 1px);
    background-size: 28px 28px;
    opacity: 0.35;
    pointer-events: none;
  }

  .snip-hint {
    position: fixed;
    left: 50%;
    top: 24px;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 10px;
    border-radius: 999px;
    padding: 9px 14px;
    font-size: 12px;
    background: rgba(7, 12, 22, 0.78);
    color: #f8fbff;
    border: 1px solid rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(10px);
  }

  .snip-size,
  .snip-size-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 24px;
    padding: 0 9px;
    border-radius: 999px;
    background: rgba(123, 224, 255, 0.16);
    color: #dff7ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.03em;
  }

  .snip-hint button {
    height: 26px;
    border-radius: 999px;
    padding: 0 10px;
    font-size: 11px;
  }

  .snip-rect {
    position: fixed;
    border-radius: 18px;
    border: 2px solid #7be0ff;
    background: rgba(255, 255, 255, 0.02);
    box-shadow:
      0 0 0 9999px rgba(5, 10, 18, 0.3),
      0 0 0 1px rgba(255, 255, 255, 0.24) inset,
      0 28px 54px rgba(0, 0, 0, 0.28);
  }

  .snip-size-badge {
    position: absolute;
    right: 10px;
    bottom: 10px;
    background: rgba(7, 12, 22, 0.78);
    border: 1px solid rgba(255, 255, 255, 0.14);
  }
</style>
