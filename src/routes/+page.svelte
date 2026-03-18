<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { getCurrentWindow } from "@tauri-apps/api/window";
  import { onDestroy, onMount } from "svelte";

  interface TranslationPayload {
    text: string;
    source_text: string;
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

  const languageOptions = [
    { label: "Japanese -> English", value: "ja-en" },
    { label: "Korean -> English", value: "ko-en" },
    { label: "Chinese (zh) -> English", value: "zh-en" },
  ];

  let mode = $state<UiMode>("control");
  const windowRole: WindowRole = getCurrentWindow().label === "overlay" ? "overlay" : "control";

  let currentLanguage = $state("ja-en");
  let modelsChecked = $state(false);
  let modelsExist = $state(false);
  let isDownloading = $state(false);
  let downloadProgress = $state<Record<string, DownloadProgress>>({});
  let downloadStartedAtMs = $state<number | null>(null);
  let lastSpeedSampleMs = $state<number | null>(null);
  let lastSpeedSampleBytes = $state(0);
  let downloadSpeedBytesPerSec = $state(0);

  let translatedText = $state("");
  let sourceText = $state("");
  let isRunning = $state(false);
  let isClickThrough = $state(false);

  let selecting = $state(false);
  let startPoint = $state<Point | null>(null);
  let selectionRect = $state<Rect | null>(null);

  let unlistenTranslation: (() => void) | null = null;
  let unlistenProgress: (() => void) | null = null;
  let unlistenOverlayMode: (() => void) | null = null;

  const expectedModelFiles = ["tokenizer.json", "encoder_model.onnx", "decoder_model_merged.onnx", "decoder_model.onnx"];

  onMount(async () => {
    await refreshModelStatus();

    if (windowRole === "overlay") {
      mode = "capture";
    } else {
      mode = "control";
    }

    unlistenTranslation = await listen<TranslationPayload>("translation-update", (event) => {
      if (windowRole !== "overlay") return;
      translatedText = event.payload.text;
      sourceText = event.payload.source_text;
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
      }
    });
  });

  onDestroy(() => {
    if (unlistenTranslation) unlistenTranslation();
    if (unlistenProgress) unlistenProgress();
    if (unlistenOverlayMode) unlistenOverlayMode();
  });

  async function refreshModelStatus() {
    modelsChecked = false;
    try {
      modelsExist = await invoke<boolean>("check_models", { languagePair: currentLanguage });
    } catch (e) {
      console.error("Model check failed:", e);
      modelsExist = false;
    } finally {
      modelsChecked = true;
    }
  }

  async function onLanguageChange(event: Event) {
    const select = event.currentTarget as HTMLSelectElement;
    currentLanguage = select.value;
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
      isDownloading = false;
      downloadSpeedBytesPerSec = 0;
      lastSpeedSampleMs = null;
      lastSpeedSampleBytes = 0;
    }
  }

  async function startCapture() {
    if (windowRole !== "control" && windowRole !== "overlay") return;

    if (!modelsExist || isDownloading) {
      alert("Install the selected language model first.");
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
      });
      await invoke("start_translation");

      isRunning = true;
      mode = "capture";
      startPoint = null;
      selectionRect = null;
    } catch (e) {
      console.error("Failed to finalize capture:", e);
      alert("Failed to finalize capture: " + e);
    }
  }

  async function cancelSnip() {
    await invoke("cancel_new_capture");
    mode = "capture";
    selecting = false;
    startPoint = null;
    selectionRect = null;
  }

  async function toggleCapturePause() {
    if (windowRole !== "overlay") return;
    if (isRunning) {
      await invoke("stop_translation");
      isRunning = false;
    } else {
      await invoke("start_translation");
      isRunning = true;
    }
  }

  async function toggleClickThrough() {
    if (windowRole !== "overlay") return;
    isClickThrough = !isClickThrough;
    await invoke("set_click_through", { enabled: isClickThrough });
  }

  async function unlockCapture() {
    if (windowRole !== "overlay") return;

    if (isRunning) {
      await invoke("stop_translation");
      isRunning = false;
    }

    await invoke("unlock_capture");
    if (isClickThrough) {
      isClickThrough = false;
      await invoke("set_click_through", { enabled: false });
    }

    translatedText = "";
    sourceText = "";
    mode = "capture";
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
</script>

<main class="surface" role="application">
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

        <div class="status-row">
          <span class="dot" class:ok={modelsExist}></span>
          <span>{modelsChecked ? (modelsExist ? "Model installed" : "Model not installed") : "Checking model..."}</span>
        </div>

        <div class="actions">
          <button onclick={installSelectedLanguage} disabled={isDownloading}>
            {isDownloading ? "Installing..." : modelsExist ? "Installed" : "Install Model"}
          </button>
          <button onclick={reinstallSelectedLanguage} disabled={isDownloading}>
            Reinstall
          </button>
          <button class="primary" onclick={startCapture} disabled={!modelsExist || isDownloading}>
            Capture
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
      <div class="capture-toolbar">
        <button onclick={toggleCapturePause}>{isRunning ? "Pause" : "Resume"}</button>
        <button onclick={toggleClickThrough}>{isClickThrough ? "Unlock UI" : "Click-through"}</button>
        <button onclick={startCapture}>Re-capture</button>
        <button class="danger" onclick={unlockCapture}>Unlock</button>
      </div>

      {#if translatedText}
        <div class="translation-in-place">{translatedText}</div>
      {/if}

      {#if sourceText}
        <div class="source-preview">{sourceText}</div>
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
        Drag to select text area
        <button onclick={cancelSnip}>Cancel</button>
      </div>

      {#if selectionRect}
        <div
          class="snip-rect"
          style="left: {selectionRect.left}px; top: {selectionRect.top}px; width: {selectionRect.width}px; height: {selectionRect.height}px;"
        ></div>
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

  .surface {
    position: fixed;
    inset: 0;
    user-select: none;
    background:
      radial-gradient(circle at 15% 15%, rgba(51, 132, 255, 0.16), transparent 42%),
      radial-gradient(circle at 80% 20%, rgba(25, 182, 255, 0.1), transparent 34%),
      linear-gradient(165deg, #0d1119, #0b1220);
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
    pointer-events: none;
  }

  .capture-toolbar {
    position: fixed;
    top: 6px;
    right: 6px;
    display: flex;
    gap: 6px;
    padding: 6px;
    border-radius: 10px;
    background: rgba(12, 14, 20, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    pointer-events: auto;
  }

  .capture-toolbar button {
    height: 28px;
    border-radius: 8px;
    padding: 0 10px;
    font-size: 12px;
  }

  .capture-toolbar .danger {
    background: rgba(255, 82, 82, 0.25);
    color: #ffd8d8;
  }

  .translation-in-place {
    position: absolute;
    left: 8px;
    right: 8px;
    bottom: 10px;
    font-size: clamp(20px, 3.8vw, 38px);
    font-weight: 700;
    line-height: 1.16;
    color: #f7fdff;
    text-shadow:
      -1px -1px 0 #000,
      1px -1px 0 #000,
      -1px 1px 0 #000,
      1px 1px 0 #000,
      0 0 11px rgba(0, 0, 0, 0.85);
    pointer-events: none;
  }

  .source-preview {
    position: absolute;
    left: 8px;
    right: 8px;
    top: 44px;
    font-size: 12px;
    color: rgba(233, 243, 255, 0.88);
    background: rgba(4, 6, 12, 0.5);
    border-radius: 8px;
    padding: 6px 8px;
    opacity: 0.85;
    pointer-events: none;
  }

  .snip-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.22);
    cursor: crosshair;
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
    padding: 8px 12px;
    font-size: 12px;
    background: rgba(10, 12, 20, 0.82);
    color: #f8fbff;
    border: 1px solid rgba(255, 255, 255, 0.2);
  }

  .snip-hint button {
    height: 24px;
    border-radius: 999px;
    padding: 0 10px;
    font-size: 11px;
  }

  .snip-rect {
    position: fixed;
    border: 2px solid #1ab6ff;
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.4);
    background: rgba(26, 182, 255, 0.1);
  }
</style>
