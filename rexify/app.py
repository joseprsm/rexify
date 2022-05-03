import os
import pandas as pd
import streamlit as st

host = st.sidebar.text_input("Kubeflow Pipelines Endpoint", "http://localhost:3000")
events_file = st.sidebar.file_uploader("Upload event data", "csv")

BASE_DIR = "/mnt/data"
OUTPUT_DIR = os.path.join(BASE_DIR, "raw")

if OUTPUT_DIR not in [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR)]:
    os.makedirs(OUTPUT_DIR)


def _trigger_pipeline():
    import kfp
    from rexify.pipeline import pipeline_fn

    kfp.Client(host=host).create_run_from_pipeline_func(pipeline_fn, arguments={})


btn = st.sidebar.button("Run", disabled=events_file is None)

if events_file is not None:

    events = pd.read_csv(events_file)
    events.columns = ["userId", "itemId"]

    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:
        st.text("Events")
        st.write(events)
        events.to_csv(os.path.join(OUTPUT_DIR, "events.csv"))

    with col2:
        st.text("Users")
        users = pd.DataFrame(events.userId.unique(), columns=["userId"])
        st.write(users)
        users.to_csv(os.path.join(OUTPUT_DIR, "users.csv"))

    with col3:
        st.text("Items")
        items = pd.DataFrame(events.itemId.unique(), columns=["itemId"])
        st.write(items)
        items.to_csv(os.path.join(OUTPUT_DIR, "items.csv"))

    if btn:
        _trigger_pipeline()
