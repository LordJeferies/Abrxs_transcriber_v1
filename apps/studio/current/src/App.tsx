import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import "./App.css";

type WorkerEvent = {
  type?: string;
  stage?: string;
  percent?: number;
  message?: string;
  code?: string;
  files?: Record<string, string>;
  jobId?: string;
};

function App() {
  const [projectId, setProjectId] = useState("JOC55_AMANDA");
  const [sourcePath, setSourcePath] = useState("");
  const [outputDir, setOutputDir] = useState("");
  const [status, setStatus] = useState("Listo");
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [files, setFiles] = useState<Record<string, string>>({});
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let dispose: (() => void) | undefined;

    listen<WorkerEvent>("worker-event", (event) => {
      const data = event.payload;

      if (data.type === "STARTED" || data.type === "ACCEPTED") {
        setRunning(true);
        setStatus(data.message || "Transcribiendo...");
      }

      if (data.type === "PROGRESS") {
        setRunning(true);
        setProgress(Number(data.percent || 0));
        setStatus(data.message || "Procesando...");
      }

      if (data.type === "LOG") {
        setLogs((previous) => [
          ...previous.slice(-99),
          `[${data.stage || "WORKER"}] ${data.message || ""}`
        ]);
      }

      if (data.type === "DONE") {
        setRunning(false);
        setProgress(100);
        setStatus("Completado");
        setFiles(data.files || {});
      }

      if (data.type === "ERROR") {
        setRunning(false);
        setStatus(`Error: ${data.message || data.code || "desconocido"}`);
      }

      if (data.type === "CANCELLED") {
        setRunning(false);
        setStatus("Cancelado");
      }

      setLogs((previous) => [
        ...previous.slice(-99),
        JSON.stringify(data)
      ]);
    }).then((fn) => {
      dispose = fn;
    });

    return () => {
      if (dispose) dispose();
    };
  }, []);

  async function selectMaster() {
    const selected = await open({
      multiple: false,
      directory: false,
      filters: [
        {
          name: "Vídeo o audio",
          extensions: [
            "mp4",
            "mov",
            "m4v",
            "mkv",
            "webm",
            "wav",
            "mp3"
          ]
        }
      ]
    });

    if (typeof selected === "string") {
      setSourcePath(selected);
      setStatus("Master seleccionado");
    }
  }

  async function selectOutput() {
    const selected = await open({
      multiple: false,
      directory: true
    });

    if (typeof selected === "string") {
      setOutputDir(selected);
    }
  }

  async function startTranscription() {
    if (!sourcePath) {
      setStatus("Selecciona un master primero");
      return;
    }

    const cleanSourcePath = sourcePath.trim();
    const cleanOutputDir = outputDir.trim();

    const destination =
      cleanOutputDir ||
      `${cleanSourcePath.substring(0, cleanSourcePath.lastIndexOf("/"))}/ABRXOS_TRANSCRIPT`;

    setOutputDir(destination);
    setProgress(0);
    setFiles({});
    setLogs([]);
    setRunning(true);
    setStatus("Iniciando Worker...");

    try {
      await invoke("start_transcription", {
        sourcePath,
        projectId,
        outputDir: destination
      });
    } catch (error) {
      setRunning(false);
      setStatus(`Error iniciando Worker: ${String(error)}`);
    }
  }

  async function openOutput() {
    if (!outputDir) {
      setStatus("No hay carpeta de salida");
      return;
    }

    try {
      await invoke("open_output", {
        path: outputDir
      });
    } catch (error) {
      setStatus(`No se pudo abrir la carpeta: ${String(error)}`);
    }
  }

  return (
    <main className="app">
      <header>
        <h1>ABRXOS Transcriber Studio</h1>
        <p>Whisper MLX · Word timing · Transcript JSON</p>
      </header>

      <section className="card">
        <h2>Proyecto</h2>

        <label>Project ID</label>
        <input
          value={projectId}
          onChange={(event) => setProjectId(event.target.value)}
        />

        <label>Master</label>
        <button onClick={selectMaster}>
          📂 Seleccionar master desde Finder
        </button>

        <div className="path">
          {sourcePath || "No se ha seleccionado ningún archivo"}
        </div>

        <label>Salida</label>
        <button onClick={selectOutput}>
          📁 Seleccionar carpeta de salida
        </button>

        <div className="path">
          {outputDir || "Se creará una carpeta automática"}
        </div>

        <div className="actions">
          <button
            className="primary"
            onClick={startTranscription}
            disabled={running}
          >
            ▶ Analizar con Whisper
          </button>

          <button
            onClick={openOutput}
            disabled={!outputDir}
          >
            Abrir salida
          </button>
        </div>
      </section>

      <section className="card">
        <h2>Estado</h2>
        <strong>{status}</strong>

        <div className="progress">
          <div style={{ width: `${progress}%` }} />
        </div>

        <span>{progress.toFixed(1)}%</span>
      </section>

      <section className="card">
        <h2>Archivos generados</h2>

        {Object.keys(files).length === 0 ? (
          <p>Aún no hay resultados.</p>
        ) : (
          Object.entries(files).map(([name, path]) => (
            <div className="file" key={name}>
              <b>{name}</b>
              <span>{path}</span>
            </div>
          ))
        )}
      </section>

      <section className="card">
        <h2>Logs</h2>
        <pre>{logs.join("\n")}</pre>
      </section>
    </main>
  );
}

export default App;
