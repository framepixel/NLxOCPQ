import os
import subprocess
import shutil
import glob
import json
import time
import signal
from urllib import request, error

def run_ocpq_query(ocel_path, query_result_file):
    """
    Executes the generated OCPQ query using the Rust CLI backend.
    Output is suppressed except for the final query result.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Resolve absolute paths
    ocel_abs = os.path.abspath(os.path.join(script_dir, ocel_path))
    query_abs = os.path.abspath(os.path.join(script_dir, query_result_file))
    
    # Path to the Rust CLI cargo project
    cli_dir = os.path.join(project_root, "OCPQ", "backend", "ocpq_cli")
    
    if not os.path.exists(cli_dir):
        warning = f"warning: OCPQ CLI directory not found at {cli_dir}"
        print(warning)
        return warning, warning, None
        
    # Find cargo executable
    cargo_cmd = shutil.which("cargo")
    if not cargo_cmd:
        # Fallback to default manual paths for Windows and Unix
        cargo_cmd_windows = os.path.expanduser("~\\.cargo\\bin\\cargo.exe")
        cargo_cmd_unix = os.path.expanduser("~/.cargo/bin/cargo")
        if os.path.exists(cargo_cmd_windows):
            cargo_cmd = cargo_cmd_windows
        elif os.path.exists(cargo_cmd_unix):
            cargo_cmd = cargo_cmd_unix
        else:
            cargo_cmd = "cargo" # final fallback
        
    # The -q (quiet) flag restricts cargo build output, only showing program output
    cmd = [
        cargo_cmd, "run", "-q", "--release", "--",
        "--ocel", ocel_abs,
        "--bbox-tree", query_abs
    ]
    
    print(f"\nExecuting OCPQ CLI query locally...")
    
    process = None
    try:
        # start_new_session=True creates a new process group so we can cleanly kill BOTH cargo and the spawned Rust executable if Streamlit stops
        process = subprocess.Popen(
            cmd, 
            cwd=cli_dir, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            start_new_session=True
        )
        
        # Wait for the process to finish
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, output=stdout, stderr=stderr
            )
            
        print("--- Query Execution Result ---")
        print(stdout.strip())
        print("-----------------------------\n")
        
        # Find the latest exported JSON file in the cli directory
        export_files = glob.glob(os.path.join(cli_dir, "ocpq-res-export-*.json"))
        latest_file = max(export_files, key=os.path.getmtime) if export_files else None
        
        # On successful execution (exit code 0), we ignore `res.stderr` (e.g. rust/cargo warnings)
        return stdout, None, latest_file
        
    except BaseException as e:
        # If the user presses "Stop" in Streamlit, it raises a custom ScriptRunner exception.
        # This BaseException catch perfectly hooks into that flow so we can kill the underlying process.
        if process and process.poll() is None:
            print("\nExecution interrupted (e.g. Streamlit Stop)! Force-killing the sub-processes...")
            try:
                # Kill the entire process group so `cargo` and its spawned `ocpq_cli` binary both die instantly
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception as kill_err:
                print(f"Failed to kill process: {kill_err}")
                
        # If it was a standard Cargo failure (e.g., syntax crash), parse it out as normal
        if isinstance(e, subprocess.CalledProcessError):
            print("--- Query Execution Failed ---")
            
            stdout_dump = e.stdout.strip() if e.stdout else ""
            stderr_dump = e.stderr.strip() if e.stderr else ""
            
            # Filter stderr to only include the panic message if it exists
            if "thread 'main'" in stderr_dump:
                stderr_dump = stderr_dump[stderr_dump.find("thread 'main'"):]
                
            if stdout_dump:
                print("Stdout Dump:\n", stdout_dump)
            if stderr_dump:
                print("Stderr Dump:\n", stderr_dump)
            print("-----------------------------\n")
            return stdout_dump, stderr_dump, None
        else:
            # Re-raise Streamlit's StopException / KeyboardInterrupt so the UI actually stops
            raise e


def _http_post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read()


def _http_post_bytes(url, payload_bytes):
    req = request.Request(url, data=payload_bytes, method="POST")
    with request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read()


def _http_get_status(url):
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=5) as resp:
        return resp.status


def _is_backend_alive(base_url):
    try:
        status = _http_get_status(base_url.rstrip("/") + "/ocel/info")
        return 200 <= status < 500
    except Exception:
        return False


def export_query_result_table(
    ocel_path,
    query_result_file,
    node_index=0,
    output_path=None,
    export_format="CSV",
    include_violation_status=True,
    include_ids=True,
    omit_header=False,
    labels=None,
    backend_url="http://127.0.0.1:3000",
    auto_start_backend=True,
):
    """
    Export OCPQ query situations as CSV/XLSX using OCPQ's built-in web-server endpoint.

    This uses the exact same backend flow as the UI:
    1) Upload OCEL to web-server state
    2) Evaluate bbox tree via /ocel/check-constraints-box
    3) Export one node's bindings via /ocel/export-bindings
    """
    labels = labels or []
    export_format = export_format.upper()
    if export_format not in {"CSV", "XLSX"}:
        raise ValueError("export_format must be 'CSV' or 'XLSX'")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    web_server_dir = os.path.join(project_root, "OCPQ", "backend", "web-server")

    ocel_abs = os.path.abspath(os.path.join(script_dir, ocel_path))
    query_abs = os.path.abspath(os.path.join(script_dir, query_result_file))

    if not os.path.exists(ocel_abs):
        raise FileNotFoundError(f"OCEL file not found: {ocel_abs}")
    if not os.path.exists(query_abs):
        raise FileNotFoundError(f"Query tree file not found: {query_abs}")

    if output_path is None:
        ext = "csv" if export_format == "CSV" else "xlsx"
        output_path = os.path.join(
            script_dir,
            f"ocpq-bindings-node-{node_index}.{ext}",
        )

    started_backend = False
    backend_proc = None
    try:
        if auto_start_backend and not _is_backend_alive(backend_url):
            if not os.path.exists(web_server_dir):
                raise FileNotFoundError(
                    f"OCPQ web-server directory not found at {web_server_dir}"
                )

            cargo_cmd = shutil.which("cargo")
            if not cargo_cmd:
                cargo_cmd_windows = os.path.expanduser("~\\.cargo\\bin\\cargo.exe")
                cargo_cmd_unix = os.path.expanduser("~/.cargo/bin/cargo")
                if os.path.exists(cargo_cmd_windows):
                    cargo_cmd = cargo_cmd_windows
                elif os.path.exists(cargo_cmd_unix):
                    cargo_cmd = cargo_cmd_unix
                else:
                    cargo_cmd = "cargo"

            cmd = [cargo_cmd, "run", "-q", "--release"]
            backend_proc = subprocess.Popen(
                cmd,
                cwd=web_server_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            started_backend = True

            for _ in range(200):
                if _is_backend_alive(backend_url):
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError(
                    f"Backend did not become ready at {backend_url} within timeout"
                )

        # 1) Upload OCEL to backend state
        if ocel_abs.endswith(".json"):
            upload_endpoint = "/ocel/upload-json"
        elif ocel_abs.endswith(".xml"):
            upload_endpoint = "/ocel/upload-xml"
        elif ocel_abs.endswith(".sqlite"):
            upload_endpoint = "/ocel/upload-sqlite"
        elif ocel_abs.endswith(".xes.gz"):
            upload_endpoint = "/ocel/upload-xes-conversion/.xes.gz"
        elif ocel_abs.endswith(".xes"):
            upload_endpoint = "/ocel/upload-xes-conversion/.xes"
        else:
            raise ValueError(
                "Unsupported OCEL file type. Use .json, .xml, .sqlite, .xes, or .xes.gz"
            )

        with open(ocel_abs, "rb") as f:
            payload_bytes = f.read()
        _http_post_bytes(backend_url.rstrip("/") + upload_endpoint, payload_bytes)

        # 2) Evaluate bbox tree (fills eval_res in backend state)
        with open(query_abs, "r", encoding="utf-8") as f:
            tree_json = json.load(f)
        _http_post_json(
            backend_url.rstrip("/") + "/ocel/check-constraints-box",
            {"tree": tree_json, "measurePerformance": False},
        )

        # 3) Export table for the selected node
        table_options = {
            "includeViolationStatus": include_violation_status,
            "includeIds": include_ids,
            "omitHeader": omit_header,
            "labels": labels,
            "format": export_format,
        }
        _status, table_bytes = _http_post_json(
            backend_url.rstrip("/") + "/ocel/export-bindings",
            [node_index, table_options],
        )

        with open(output_path, "wb") as f:
            f.write(table_bytes)
        return os.path.abspath(output_path)

    except error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP error calling OCPQ backend: {e.code} {e.reason}. Response: {body}"
        ) from e
    finally:
        if started_backend and backend_proc is not None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=3)
            except Exception:
                backend_proc.kill()
