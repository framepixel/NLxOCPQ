import streamlit as st
import os
import glob
import time
from LLM import set_hf_client, nl_to_ocpq_pipeline, dataset, embeddings

def main():
    st.set_page_config(page_title="NL to OCPQ Generator", layout="wide")
    st.title("Natural Language to Object-Centric Process Query (OCPQ)")

    # Provide HF Token in UI or env
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = st.sidebar.text_input("Hugging Face API Token", type="password")
        if not hf_token:
            st.warning("Please provide a Hugging Face API Token (HF_TOKEN) in the sidebar or environment variables to proceed.")
            st.stop()
    
    set_hf_client(hf_token)

    st.sidebar.header("Configuration")
    
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
                    user_query, dataset, embeddings, "test_query.json", selected_file
                )
                
                # Create tabs for structured outputs instead of a giant vertical scrolling log
                tab1, tab2, tab3 = st.tabs(["Generation Pipeline", "CLI Output", "Data Preview"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("1. Extracted Structure")
                        if structure:
                            st.json(structure)
                        else:
                            st.warning("Failed to parse structure json.")
                    with col2:
                        st.subheader("2. Generated Query Tree")
                        if json_tree:
                            st.json(json_tree)
                        else:
                            st.warning("Failed to parse query tree json.")
                    
                    st.subheader("3. Final OCPQ JSON")
                    if ocpq_json:
                        st.json(ocpq_json)
                    else:
                        st.warning("Failed to parse Final OCPQ json.")
                    
                with tab2:
                    st.subheader("Query Execution Logs")
                    if query_result:
                        st.code(query_result, language="text")
                    else:
                        st.warning("No result returned from Query Executor. Check CLI logs.")

                with tab3:
                    st.subheader("Resulting Execution Output")
                    if query_result_file_path and os.path.exists(query_result_file_path):
                        st.success(f"Results exported successfully to `{query_result_file_path}`!")
                        try:
                            import json
                            with open(query_result_file_path, "r", encoding="utf-8") as res_file:
                                result_data = json.load(res_file)
                                
                            if "Ok" in result_data:
                                eval_data = result_data["Ok"].get("evaluationResults", [])
                                if isinstance(eval_data, list) and len(eval_data) > 0:
                                    st.write(f"Evaluated Items: **{len(eval_data)}**")
                                    # We flatten or just use st.json for now to preview complex nested records
                                    st.json(eval_data)
                                else:
                                    st.write("Query ran successfully, but no situations matched.")
                                    st.json(result_data)
                            elif "Err" in result_data:
                                st.error("The backend returned an error in the JSON structure.")
                                st.json(result_data["Err"])
                            else:
                                st.json(result_data)
                                
                        except Exception as parse_e:
                            st.warning(f"Could not parse the result file as JSON: {parse_e}")
                    else:
                        st.info("No formatted data output file was captured during this run.")

            except Exception as e:
                st.error(f"An error occurred during API execution: {str(e)}")

if __name__ == "__main__":
    main()
