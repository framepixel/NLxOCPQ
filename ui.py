import os
import glob
import time
import warnings
import logging
import json

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st

warnings.filterwarnings(
    "ignore",
    message=r"Accessing `__path__` from .*Returning `__path__` instead\. Behavior may be different and this alias will be removed in future versions\.",
)

from LLM import (
    set_hf_client,
    nl_to_ocpq_pipeline,
    dataset,
    embeddings,
    PROMPT_STRATEGIES,
    DEFAULT_PROMPT_STRATEGY,
)


RAW_JSON_RENDER_LIMIT_BYTES = 5 * 1024 * 1024
RESULT_FILE_LARGE_MODE_BYTES = 50 * 1024 * 1024


def _format_violation(violation):
    if violation is None:
        return "Satisfied"
    if isinstance(violation, str):
        return f"Violated: {violation}"
    if isinstance(violation, dict) and len(violation) == 1:
        k, v = next(iter(violation.items()))
        return f"Violated: {k} ({v})"
    return f"Violated: {violation}"


def _readable_rows_for_node(node_data, object_ids, event_ids, max_rows=200):
    rows = []
    situations = node_data.get("situations", []) if isinstance(node_data, dict) else []
    for i, situation in enumerate(situations[:max_rows], start=1):
        if not isinstance(situation, list) or len(situation) != 2:
            continue
        binding, violation = situation
        event_map = binding.get("eventMap", []) if isinstance(binding, dict) else []
        object_map = binding.get("objectMap", []) if isinstance(binding, dict) else []
        label_map = binding.get("labelMap", []) if isinstance(binding, dict) else []

        event_parts = []
        for ev_var, ev_idx in event_map:
            ev_name = event_ids[ev_idx] if isinstance(ev_idx, int) and 0 <= ev_idx < len(event_ids) else f"index:{ev_idx}"
            event_parts.append(f"e{ev_var + 1}={ev_name}")

        object_parts = []
        for ob_var, ob_idx in object_map:
            ob_name = object_ids[ob_idx] if isinstance(ob_idx, int) and 0 <= ob_idx < len(object_ids) else f"index:{ob_idx}"
            object_parts.append(f"o{ob_var + 1}={ob_name}")

        label_parts = []
        for item in label_map:
            if isinstance(item, list) and len(item) == 2:
                label_parts.append(f"{item[0]}={item[1]}")

        rows.append(
            {
                "Situation": i,
                "Status": _format_violation(violation),
                "Events": ", ".join(event_parts) if event_parts else "-",
                "Objects": ", ".join(object_parts) if object_parts else "-",
                "Labels": ", ".join(label_parts) if label_parts else "-",
            }
        )
    return rows, len(situations)


def render_human_readable_result(result_data, allow_raw_json=False):
    if "Err" in result_data:
        st.error("The backend returned an error in the JSON structure.")
        err_data = result_data["Err"]
        if allow_raw_json:
            st.json(err_data)
        else:
            st.code(str(err_data), language="text")
        return

    if "Ok" not in result_data:
        st.warning("Unexpected result shape. Showing raw JSON.")
        if allow_raw_json:
            st.json(result_data)
        else:
            st.code(str(result_data), language="text")
        return

    ok_data = result_data.get("Ok", {})
    eval_data = ok_data.get("evaluationResults", [])
    object_ids = ok_data.get("objectIds", [])
    event_ids = ok_data.get("eventIds", [])
    bindings_skipped = ok_data.get("bindingsSkipped", False)

    if not isinstance(eval_data, list) or len(eval_data) == 0:
        st.info("Query ran successfully, but no evaluation results were produced.")
        return

    total_situations = 0
    total_violations = 0
    node_summary = []
    for idx, node in enumerate(eval_data):
        count = node.get("situationCount", 0)
        violated = node.get("situationViolatedCount", 0)
        total_situations += count
        total_violations += violated
        node_summary.append(
            {
                "Node": idx,
                "Situation Count": count,
                "Violated Count": violated,
                "Satisfaction Rate": f"{(100.0 * (count - violated) / count):.1f}%" if count else "-",
            }
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", len(eval_data))
    c2.metric("Total Situations", total_situations)
    c3.metric("Total Violations", total_violations)
    c4.metric("Bindings Skipped", "Yes" if bindings_skipped else "No")

    st.subheader("Node Overview")
    st.dataframe(node_summary, width="stretch")

    st.subheader("Situation Details")
    selected_node = st.selectbox(
        "Choose node",
        options=list(range(len(eval_data))),
        format_func=lambda x: f"Node {x} (count={eval_data[x].get('situationCount', 0)}, violated={eval_data[x].get('situationViolatedCount', 0)})",
    )
    max_rows = st.slider("Max rows to preview", min_value=20, max_value=1000, value=200, step=20)

    rows, total_for_node = _readable_rows_for_node(
        eval_data[selected_node],
        object_ids,
        event_ids,
        max_rows=max_rows,
    )

    if total_for_node == 0:
        st.info("No situations matched for this node.")
    elif rows:
        st.caption(f"Showing {len(rows)} of {total_for_node} situations for node {selected_node}.")
        st.dataframe(rows, width="stretch")
    else:
        st.info("No readable rows were generated for this node. Showing raw node JSON.")
        if allow_raw_json:
            st.json(eval_data[selected_node])
        else:
            st.code(str(eval_data[selected_node]), language="text")

    if allow_raw_json:
        with st.expander("Raw result JSON"):
            st.json(result_data)
    else:
        st.caption(
            "Raw JSON preview disabled for large result files to avoid Streamlit websocket size errors."
        )

def main():
    st.set_page_config(page_title="NL to OCPQ Generator", layout="wide")
    st.title("Natural Language to Object-Centric Process Query (OCPQ)")

    if "last_pipeline_output" not in st.session_state:
        st.session_state["last_pipeline_output"] = None
    if "last_pipeline_error" not in st.session_state:
        st.session_state["last_pipeline_error"] = None

    # Provide HF Token in UI or env
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = st.sidebar.text_input("Hugging Face API Token", type="password")
        if not hf_token:
            st.warning("Please provide a Hugging Face API Token (HF_TOKEN) in the sidebar or environment variables to proceed.")
            st.stop()
    
    set_hf_client(hf_token)

    st.sidebar.header("Configuration")
    selected_strategy = st.sidebar.selectbox(
        "Prompting Strategy",
        options=PROMPT_STRATEGIES,
        index=PROMPT_STRATEGIES.index(DEFAULT_PROMPT_STRATEGY)
        if DEFAULT_PROMPT_STRATEGY in PROMPT_STRATEGIES
        else 0,
    )
    
    st.sidebar.subheader("Event Log Selection")
    uploaded_file = st.sidebar.file_uploader("Upload custom OCEL JSON Log", type=["json"])
    
    if uploaded_file is not None:
        os.makedirs("uploaded_logs", exist_ok=True)
        selected_file = os.path.join("uploaded_logs", uploaded_file.name).replace("\\", "/")
        with open(selected_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Using uploaded log: {uploaded_file.name}")
    else:
        # Simple relative path globbing matching original script assumptions
        json_files = glob.glob("../*.json")
        if not json_files:
            st.sidebar.warning("No JSON log files found in the parent directory.")
            selected_file = None
        else:
            selected_file = st.sidebar.selectbox("Or select existing OCEL Event Log", json_files)

    with st.form("query_form"):
        user_query = st.text_area(
            "Enter your natural language prompt:",
            "For any person, the diagnosis must be completed within 3 days after the registration at ER."
        )
        
        submit_button = st.form_submit_button("Generate & Execute Query")

    if submit_button:
        if not selected_file:
            st.error("Please select an OCEL file first.")
            return
            
        with st.spinner("Generating and executing query (this may take a moment)..."):
            try:
                structure, json_tree, ocpq_json, query_result, query_result_file_path = nl_to_ocpq_pipeline(
                    user_query,
                    dataset,
                    embeddings,
                    "test_query.json",
                    selected_file,
                    prompt_strategy=selected_strategy,
                )
                st.session_state["last_pipeline_output"] = {
                    "structure": structure,
                    "json_tree": json_tree,
                    "ocpq_json": ocpq_json,
                    "query_result": query_result,
                    "query_result_file_path": query_result_file_path,
                }
                st.session_state["last_pipeline_error"] = None

            except Exception as e:
                st.session_state["last_pipeline_error"] = str(e)

    if st.session_state.get("last_pipeline_error"):
        st.error(f"An error occurred during API execution: {st.session_state['last_pipeline_error']}")

    last_output = st.session_state.get("last_pipeline_output")
    if last_output:
        # Create tabs for structured outputs instead of a giant vertical scrolling log
        tab1, tab2, tab3 = st.tabs(["Generation Pipeline", "CLI Output", "Data Preview"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("1. Extracted Structure")
                if last_output["structure"]:
                    st.json(last_output["structure"])
                else:
                    st.warning("Failed to parse structure json.")
            with col2:
                st.subheader("2. Generated Query Tree")
                if last_output["json_tree"]:
                    st.json(last_output["json_tree"])
                else:
                    st.warning("Failed to parse query tree json.")

            st.subheader("3. Final OCPQ JSON")
            if last_output["ocpq_json"]:
                st.json(last_output["ocpq_json"])
            else:
                st.warning("Failed to parse Final OCPQ json.")

        with tab2:
            st.subheader("Query Execution Logs")
            if last_output["query_result"]:
                st.code(last_output["query_result"], language="text")
            else:
                st.warning("No result returned from Query Executor. Check CLI logs.")

        with tab3:
            st.subheader("Resulting Execution Output")
            query_result_file_path = last_output["query_result_file_path"]
            if query_result_file_path and os.path.exists(query_result_file_path):
                st.success(f"Results exported successfully to `{query_result_file_path}`!")
                try:
                    result_file_size = os.path.getsize(query_result_file_path)
                    large_result_mode = result_file_size > RESULT_FILE_LARGE_MODE_BYTES

                    st.caption(
                        f"Result file size: {result_file_size / (1024 * 1024):.1f} MB"
                    )
                    if large_result_mode:
                        st.warning(
                            "Large result mode is enabled. Raw JSON views are disabled to avoid websocket message size overflows."
                        )

                    with open(query_result_file_path, "r", encoding="utf-8") as res_file:
                        result_data = json.load(res_file)

                    allow_raw_json = result_file_size <= RAW_JSON_RENDER_LIMIT_BYTES
                    render_human_readable_result(result_data, allow_raw_json=allow_raw_json)
                except Exception as parse_e:
                    st.warning(f"Could not parse the result file as JSON: {parse_e}")
            else:
                st.info("No formatted data output file was captured during this run.")

if __name__ == "__main__":
    main()
