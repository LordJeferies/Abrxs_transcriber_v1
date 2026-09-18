#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde_json::{json, Value};
use std::fs::{create_dir_all, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use tauri::{AppHandle, Emitter};

const WORKER_PATH: &str = "/Users/lordjef/Downloads/Abrxs_transcriber_v1/apps/worker/current/worker.py";
const LOG_DIR: &str = "/Users/lordjef/Downloads/Abrxs_transcriber_v1/private/tauri-logs";

fn append_log(path: &PathBuf, text: &str) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{}", text);
        let _ = file.flush();
    }
}

fn open_terminal_log(log_path: &PathBuf) {
    let quoted = log_path.to_string_lossy().replace("\"", "\\\"");

    let script = format!(
        "tell application \"Terminal\" to activate\n\
         tell application \"Terminal\" to do script \"echo 'ABRXOS TRANSCRIBER PROCESS'; echo 'Log: {0}'; echo ''; tail -f '{0}'\"",
        quoted
    );

    let _ = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .spawn();
}

#[tauri::command]
fn start_transcription(
    app: AppHandle,
    source_path: String,
    project_id: String,
    output_dir: String,
) -> Result<(), String> {
    create_dir_all(LOG_DIR)
        .map_err(|error| format!("No se pudo crear el directorio de logs: {error}"))?;

    let job_id = format!(
        "TAURI_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_secs()
    );

    let log_path = PathBuf::from(LOG_DIR).join(format!("{job_id}.log"));

    append_log(&log_path, "==================================================");
    append_log(&log_path, "ABRXOS TRANSCRIBER STUDIO");
    append_log(&log_path, &format!("JOB: {job_id}"));
    append_log(&log_path, &format!("SOURCE: {source_path}"));
    append_log(&log_path, &format!("PROJECT: {project_id}"));
    append_log(&log_path, &format!("OUTPUT: {output_dir}"));
    append_log(&log_path, "Iniciando Python Worker...");

    open_terminal_log(&log_path);

    let mut child = Command::new("python3")
        .arg("-u")
        .arg(WORKER_PATH)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            append_log(&log_path, &format!("ERROR iniciando Worker: {error}"));
            format!("No se pudo iniciar Python Worker: {error}")
        })?;

    let command = json!({
        "type": "TRANSCRIBE",
        "jobId": job_id,
        "sourcePath": source_path.trim(),
        "projectId": project_id,
        "sourceId": "DEFAULT",
        "outputDir": output_dir.trim(),
        "model": "mlx-community/whisper-large-v3-turbo",
        "language": "es"
    });

    let line = serde_json::to_string(&command)
        .map_err(|error| error.to_string())?;

    append_log(&log_path, &format!("COMMAND: {line}"));

    let mut stdin = child
        .stdin
        .take()
        .ok_or("No se pudo abrir stdin del Worker")?;

    stdin
        .write_all(format!("{line}\n").as_bytes())
        .map_err(|error| error.to_string())?;

    stdin.flush().map_err(|error| error.to_string())?;

    let stdout = child
        .stdout
        .take()
        .ok_or("No se pudo abrir stdout del Worker")?;

    let stderr = child
        .stderr
        .take()
        .ok_or("No se pudo abrir stderr del Worker")?;

    let app_stdout = app.clone();
    let stdout_log = log_path.clone();

    thread::spawn(move || {
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let Ok(line) = line else {
                continue;
            };

            append_log(&stdout_log, &format!("[STDOUT] {line}"));

            if let Ok(event) = serde_json::from_str::<Value>(&line) {
                let _ = app_stdout.emit("worker-event", event);
            } else {
                let _ = app_stdout.emit(
                    "worker-event",
                    json!({
                        "type": "LOG",
                        "stage": "STDOUT",
                        "message": line
                    }),
                );
            }
        }
    });

    let app_stderr = app.clone();
    let stderr_log = log_path.clone();

    thread::spawn(move || {
        let reader = BufReader::new(stderr);

        for line in reader.lines() {
            let Ok(line) = line else {
                continue;
            };

            append_log(&stderr_log, &format!("[STDERR] {line}"));

            let _ = app_stderr.emit(
                "worker-event",
                json!({
                    "type": "LOG",
                    "stage": "STDERR",
                    "message": line
                }),
            );
        }
    });

    let wait_log = log_path.clone();

    thread::spawn(move || {
        let result = child.wait();

        match result {
            Ok(status) => append_log(
                &wait_log,
                &format!("WORKER TERMINÓ: {status}"),
            ),
            Err(error) => append_log(
                &wait_log,
                &format!("ERROR esperando Worker: {error}"),
            ),
        }

        drop(stdin);
    });

    Ok(())
}

#[tauri::command]
fn open_output(path: String) -> Result<(), String> {
    Command::new("open")
        .arg(path)
        .spawn()
        .map_err(|error| error.to_string())?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            start_transcription,
            open_output
        ])
        .run(tauri::generate_context!())
        .expect("error while running ABRXOS Transcriber Studio");
}
