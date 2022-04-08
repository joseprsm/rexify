import os
import subprocess

import pandas as pd
import streamlit as st

host = st.sidebar.text_input('Kubeflow Pipelines Endpoint', 'http://localhost:3000')
events_file = st.sidebar.file_uploader('Upload event data', 'csv')

OUTPUT_DIR = os.path.join('rexify', 'data')
EVENT_PATH = os.path.join(OUTPUT_DIR, 'events')
USER_PATH = os.path.join(OUTPUT_DIR, 'users')
ITEM_PATH = os.path.join(OUTPUT_DIR, 'items')


def _trigger_pipeline():
    import kfp
    subprocess.run(['python', 'rexify/runner.py'])
    kfp.Client(host=host).create_run_from_pipeline_package('rexify.tar.gz', {})


btn = st.sidebar.button('Run', disabled=events_file is None)

if events_file is not None:

    events = pd.read_csv(events_file)
    events.columns = ['userId', 'itemId']

    col1, col2, col3 = st.columns([1.5, 1, 1])
    dir_list = os.listdir(OUTPUT_DIR)

    with col1:
        st.text('Events')
        st.write(events)
        if 'events' not in dir_list:
            os.makedirs(EVENT_PATH)
        events.to_csv(os.path.join(EVENT_PATH, 'events.csv'))

    with col2:
        st.text('Users')
        users = pd.DataFrame(events.userId.unique(), columns=['userId'])
        st.write(users)
        if 'users' not in dir_list:
            os.makedirs(USER_PATH)
        users.to_csv(os.path.join(USER_PATH, 'users.csv'))

    with col3:
        st.text('Items')
        items = pd.DataFrame(events.itemId.unique(), columns=['itemId'])
        st.write(items)
        if 'items' not in dir_list:
            os.makedirs(ITEM_PATH)
        items.to_csv(os.path.join(ITEM_PATH, 'items.csv'))

    if btn:
        _trigger_pipeline()
