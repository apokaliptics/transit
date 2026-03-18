<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount, onDestroy } from "svelte";

  interface TranslationPayload {
    text: string;
    source_text: string;
  }

  let translatedText = $state("");
  let sourceText = $state("");
  let isRunning = $state(false);
  let isClickThrough = $state(false);
  let showControls = $state(false);
  let unlisten_translation: (() => void) | null = null;
  let unlisten_progress: (() => void) | null = null;

  // Downloader state
  let currentLanguage = "ja-en"; // Defaulting to ja-en for now
  let modelsChecked = $state(false);
  let modelsExist = $state(false);
  let isDownloading = $state(false);
  let downloadProgress = $state<{
    [key: string]: { downloaded: number; total: number | null };
  }>({});

  onMount(async () => {
    // Check if models exist
    try {
      modelsExist = await invoke("check_models", { languagePair: currentLanguage });
      if (modelsExist) {
        // Auto-initialized in lib.rs setup if it's ja-en, but safe to init_engine again
        await invoke("init_engine", { languagePair: currentLanguage });
      }
    } catch (e) {
      console.error("Failed to check models:", e);
    }
    modelsChecked = true;

    // Listen for translation events from the Rust backend
    unlisten_translation = await listen<TranslationPayload>("translation-update", (event) => {
      translatedText = event.payload.text;
      sourceText = event.payload.source_text;
    });

    // Listen for download progress
    unlisten_progress = await listen<any>("download-progress", (event) => {
      const p = event.payload;
      downloadProgress[p.file] = { downloaded: p.downloaded, total: p.total };
    });
  });

  onDestroy(() => {
    if (unlisten_translation) unlisten_translation();
    if (unlisten_progress) unlisten_progress();
  });

  async function startDownload() {
    isDownloading = true;
    try {
      await invoke("download_models", { languagePair: currentLanguage });
      await invoke("init_engine", { languagePair: currentLanguage });
      modelsExist = true;
    } catch (e) {
      console.error("Failed to download models:", e);
      alert("Failed to download models: " + e);
    } finally {
      isDownloading = false;
    }
  }

  async function toggleTranslation() {
    if (isRunning) {
      await invoke("stop_translation");
      isRunning = false;
      translatedText = "";
      sourceText = "";
    } else {
      await invoke("start_translation");
      isRunning = true;
    }
  }

  async function toggleClickThrough() {
    isClickThrough = !isClickThrough;
    await invoke("set_click_through", { enabled: isClickThrough });
  }
</script>

<main
  class="overlay"
  role="application"
  onmouseenter={() => (showControls = true)}
  onmouseleave={() => (showControls = false)}
>
  <!-- Setup / Download Overlay -->
  {#if modelsChecked && !modelsExist}
    <div class="setup-overlay">
      <div class="setup-card" data-tauri-drag-region>
        <h2>Model Setup Required</h2>
        <p>To perform offline translation, the app needs to download the neural translation models (~150MB).</p>
        
        {#if isDownloading}
          <div class="progress-section">
            {#each Object.entries(downloadProgress) as [file, p]}
              <div class="progress-item">
                <div class="progress-info">
                  <span class="filename">{file}</span>
                  <span class="bytes">
                    {(p.downloaded / 1024 / 1024).toFixed(1)}MB 
                    {#if p.total}
                       / {(p.total / 1024 / 1024).toFixed(1)}MB
                    {/if}
                  </span>
                </div>
                {#if p.total}
                  <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {(p.downloaded / p.total) * 100}%"></div>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {:else}
          <button class="download-btn" onclick={startDownload}>Download Models</button>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Translated subtitle text -->
  {#if modelsExist && translatedText}
    <div class="subtitle-container">
      <p class="subtitle">{translatedText}</p>
    </div>
  {/if}

  <!-- Minimal control bar — visible on hover -->
  <div class="controls" class:visible={showControls && !isClickThrough}>
    <!-- Drag region for moving the window -->
    <div class="drag-region" data-tauri-drag-region></div>

    <div class="control-buttons">
      <button
        class="control-btn"
        class:active={isRunning}
        onclick={toggleTranslation}
        title={isRunning ? "Stop Translation" : "Start Translation"}
      >
        {isRunning ? "⏹" : "▶"}
      </button>

      <button
        class="control-btn"
        class:active={isClickThrough}
        onclick={toggleClickThrough}
        title={isClickThrough ? "Disable Click-Through" : "Enable Click-Through"}
      >
        {isClickThrough ? "🔓" : "🔒"}
      </button>
    </div>

    <!-- Status indicator -->
    <div class="status">
      {#if modelsExist}
        <span class="status-dot" class:running={isRunning}></span>
        <span class="status-text">
          {isRunning ? "Translating..." : "Paused"}
        </span>
      {:else}
        <span class="status-dot" style="background: #f59e0b"></span>
        <span class="status-text">Setup Required</span>
      {/if}
    </div>
  </div>
</main>

<style>
  :global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  :global(html),
  :global(body) {
    background: transparent;
    overflow: hidden;
    width: 100%;
    height: 100%;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  }

  .overlay {
    position: fixed;
    inset: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-items: center;
    background: transparent;
    user-select: none;
  }

  /* === Subtitle Display === */
  .subtitle-container {
    width: 100%;
    padding: 8px 16px 12px;
    display: flex;
    justify-content: center;
    background: linear-gradient(
      to top,
      rgba(0, 0, 0, 0.7) 0%,
      rgba(0, 0, 0, 0.4) 60%,
      transparent 100%
    );
  }

  .subtitle {
    font-size: 22px;
    font-weight: 600;
    color: #ffffff;
    text-align: center;
    line-height: 1.4;
    max-width: 90%;
    text-shadow:
      -1px -1px 0 #000,
       1px -1px 0 #000,
      -1px  1px 0 #000,
       1px  1px 0 #000,
       0 0 8px rgba(0, 0, 0, 0.8),
       0 0 16px rgba(0, 0, 0, 0.5);
    letter-spacing: 0.02em;
    word-wrap: break-word;
  }

  /* === Controls === */
  .controls {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    background: rgba(15, 15, 15, 0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    opacity: 0;
    transform: translateY(-100%);
    transition: all 0.25s ease;
    pointer-events: none;
    z-index: 100;
  }

  .controls.visible {
    opacity: 1;
    transform: translateY(0);
    pointer-events: all;
  }

  .drag-region {
    flex: 1;
    height: 28px;
    cursor: grab;
    -webkit-app-region: drag;
  }

  .control-buttons {
    display: flex;
    gap: 4px;
  }

  .control-btn {
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.08);
    color: #ccc;
    font-size: 12px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
    -webkit-app-region: no-drag;
  }

  .control-btn:hover {
    background: rgba(255, 255, 255, 0.15);
    color: #fff;
  }

  .control-btn.active {
    background: rgba(56, 189, 248, 0.2);
    color: #38bdf8;
  }

  /* === Status === */
  .status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 8px;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #666;
    transition: background 0.3s ease;
  }

  .status-dot.running {
    background: #34d399;
    box-shadow: 0 0 6px rgba(52, 211, 153, 0.5);
    animation: pulse 2s infinite;
  }

  .status-text {
    font-size: 11px;
    color: #888;
    white-space: nowrap;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }

  /* === Setup Overlay === */
  .setup-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 200;
  }

  .setup-card {
    background: rgba(20, 20, 20, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    border-radius: 12px;
    padding: 24px;
    width: 90%;
    max-width: 500px;
    color: #fff;
    cursor: grab;
  }

  .setup-card h2 {
    font-size: 18px;
    margin-bottom: 8px;
    font-weight: 600;
    -webkit-app-region: no-drag;
  }

  .setup-card p {
    font-size: 13px;
    color: #aaa;
    margin-bottom: 20px;
    line-height: 1.5;
    -webkit-app-region: no-drag;
  }

  .download-btn {
    width: 100%;
    padding: 10px;
    background: #38bdf8;
    color: #000;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
    -webkit-app-region: no-drag;
  }

  .download-btn:hover {
    background: #0ea5e9;
  }

  .progress-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
    -webkit-app-region: no-drag;
  }

  .progress-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
  }

  .filename {
    color: #ddd;
    font-weight: 500;
  }

  .bytes {
    color: #888;
  }

  .progress-bar-bg {
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-bar-fill {
    height: 100%;
    background: #38bdf8;
    transition: width 0.1s linear;
  }
</style>
