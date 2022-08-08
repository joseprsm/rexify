import json
import os

import requests
import pandas as pd
import streamlit as st

from rexify.pipeline import compile_

events = None
with open("demo/datasets.json", "r") as f:
    datasets = json.load(f)

data_tab, schema_tab, pipeline_tab = st.tabs(["Data", "Schema", "Pipeline"])

with data_tab:

    event_url = st.text_input("Input data URL")

    st.write("Or select one of these datasets:")

    for i, col in enumerate(st.columns(len(datasets))):
        if col.button(list(datasets.keys())[i]):
            event_url = list(datasets.values())[i]

    if st.button("Download"):
        response = requests.get(event_url)
        with open("events.csv", "w") as f:
            f.write(response.text)

    if "events.csv" in os.listdir("."):
        events = pd.read_csv("events.csv")

    if events is not None:
        st.dataframe(events)

with schema_tab:
    schema = dict(user=dict(), item=dict())
    upload_schema_disabled = True

    if events is not None:

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("User features"):
                user_feats = st.multiselect("User features", events.columns)

                for feat in user_feats:
                    st.write(feat)
                    option = st.selectbox(
                        "Data type",
                        ("id", "categorical", "numerical", "timestamp"),
                        key=f"id_{feat}",
                    )
                    schema["user"][feat] = option

            with st.expander("Item features"):
                item_feats = st.multiselect("Item features", events.columns)

                for feat in item_feats:
                    st.write(feat)
                    option = st.selectbox(
                        "Data type",
                        ("id", "categorical", "numerical", "timestamp"),
                        key=f"id_{feat}",
                    )
                    schema["item"][feat] = option

        with col2:
            st.code(json.dumps(schema, indent=2))


with pipeline_tab:

    epochs = st.slider("Epochs", 1, 100)

    if (event_url is not None) and (schema is not None):
        if st.button(
            "Compile pipeline",
            on_click=compile_(event_uri=event_url, schema=schema, epochs=epochs),
        ):

            with open("pipeline.json") as f:
                pipeline_spec = json.load(f)

            with st.expander("Pipeline Spec"):
                st.code(json.dumps(pipeline_spec, indent=2))

            st.download_button(
                "Download Pipeline Spec",
                json.dumps(pipeline_spec, indent=2),
                "pipeline.json",
            )
