import json

import pandas as pd
import streamlit as st

from rexify.pipeline import compile_

events = schema_uri = None

data_tab, schema_tab, pipeline_tab = st.tabs(["Data", "Schema", "Pipeline"])

with data_tab:

    uploaded_file = st.file_uploader('Upload events data')
    if uploaded_file:
        events = pd.read_csv(uploaded_file)

    events_uri = st.text_input('Event data URL')
    st.button('Upload')


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

        if (len(user_feats) != 0) and (len(item_feats) != 0):
            upload_schema_disabled = False

        schema_uri = st.text_input("Schema upload location")
        st.button("Upload schema", disabled=upload_schema_disabled)

with pipeline_tab:

    if (events_uri is not None) and (schema_uri is not None):
        if st.button(
            "Compile pipeline",
            on_click=compile_(event_uri=events_uri, schema_uri=schema_uri),
        ):

            with open("pipeline.json") as f:
                pipeline_spec = json.load(f)

            with st.expander('Pipeline Spec'):
                st.code(json.dumps(pipeline_spec, indent=2))
