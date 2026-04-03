import os
import subprocess
import shutil

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
        print(f"Warning: OCPQ CLI directory not found at {cli_dir}")
        return None
        
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
    try:
        res = subprocess.run(cmd, cwd=cli_dir, capture_output=True, text=True, check=True)
        print("--- Query Execution Result ---")
        print(res.stdout.strip())
        print("-----------------------------\n")
        return res.stdout
    except subprocess.CalledProcessError as e:
        print("--- Query Execution Failed ---")
        if e.stdout.strip():
            print("Stdout Dump:\n", e.stdout.strip())
        if e.stderr.strip():
            print("Stderr Dump:\n", e.stderr.strip())
        print("-----------------------------\n")
        return None
