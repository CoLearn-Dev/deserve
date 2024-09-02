import pickle
import time
from typing import Any, Optional

import altair as alt
import numpy as np
import pandas as pd  # type: ignore
import requests
import streamlit as st
import torch
from streamlit.delta_generator import DeltaGenerator
from transformers import AutoTokenizer  # type: ignore

from deserve_utils.serde import dumps, loads
from deserve_worker.trace import OpId


def refresh_tokens(
    messages: list[dict[str, str]], generator: DeltaGenerator, index: int
) -> None:
    response = requests.post(
        "http://localhost:19000/chat",
        json={
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "messages": messages,
        },
        stream=True,
    )
    content = ""
    for chunk in response.iter_content():
        if chunk:
            content += chunk.decode("utf-8")
            generator.markdown(content)
    st.session_state.messages[index]["content"] = content


def request_traces(messages: list[dict[str, str]]) -> tuple[
    Optional[str],
    dict[str, torch.Tensor],
    list[tuple[int, float]],
    dict[OpId, list[OpId]],
]:
    response = requests.post(
        "http://localhost:19000/trace_chat",
        json={"model": "meta-llama/Meta-Llama-3-70B-Instruct", "messages": messages},
    )
    tensors = {}
    output2input = {}
    next_token = None
    probs = []
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            temp_tensors, metadata = loads(chunk)
            if "token" in metadata:
                next_token = metadata["token"]
            if "probs" in metadata:
                probs.extend(metadata["probs"])
            tensors.update(temp_tensors)
            str_output2input = metadata["output2input"]
            temp_output2input = {
                OpId.from_str(k): [OpId.from_str(i) for i in v]
                for k, v in str_output2input.items()
            }
            output2input.update(temp_output2input)
    return next_token, tensors, probs, output2input


@st.dialog("Select the suspicious token", width="large")
def select_challenge_token(index: int) -> None:
    st.write("Click on the token you think is suspicious")
    content = st.session_state.messages[index]["content"]
    token_ids = st.session_state.tokenizer.encode(content, add_special_tokens=False)
    tokens = [
        st.session_state.tokenizer.decode([token_id], truncate_at_eos=False)
        for token_id in token_ids
    ]
    cols = st.columns(6)
    for i, token in enumerate(tokens):
        with cols[i % 6]:
            if st.button(
                token,
                key=f"challenge_{index}_{i}",
                use_container_width=True,
            ):
                st.session_state.selected_token = (
                    index,
                    sum([len(t) for t in tokens[:i]]),
                    token,
                )
                if "traces" in st.session_state:
                    st.session_state.pop("traces")
                if "diffs" in st.session_state:
                    st.session_state.pop("diffs")
                if "output2input" in st.session_state:
                    st.session_state.pop("output2input")
                if "fraud_proof" in st.session_state:
                    st.session_state.pop("fraud_proof")
                st.switch_page(page_verify),


def render_msg(msg: dict[str, str], index: int) -> DeltaGenerator:
    with st.chat_message(msg["role"]):
        content = st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if st.button(
                "Challenge correctness",
                type="primary",
                key=f"challenge_{index}",
                use_container_width=True,
            ):
                select_challenge_token(index)
        return content


def f_page_chat() -> None:
    # ref: https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
    st.title("Example LLM Application")

    # init
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_token" not in st.session_state:
        st.session_state.selected_token = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-70B-Instruct"
        )

    for index, message in enumerate(st.session_state.messages):
        render_msg(message, index)
    if prompt := st.chat_input("Prompt to query the model here..."):
        new_user_msg = {
            "role": "user",
            "content": prompt,
        }
        render_msg(new_user_msg, len(st.session_state.messages))
        st.session_state.messages.append(new_user_msg)

        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages
        ]

        # Display assistant response in chat message container
        new_assistant_msg = {
            "role": "assistant",
            "content": "",
        }
        generator = render_msg(new_assistant_msg, len(st.session_state.messages))
        st.session_state.messages.append(new_assistant_msg)
        refresh_tokens(messages, generator, len(st.session_state.messages) - 1)

        # Add assistant response to chat history


page_chat = st.Page(
    f_page_chat, title="Example LLM Application", icon=":material/favorite:"
)


def f_page_verify() -> None:
    st.title("Inference Result Verification")

    if "selected_token" not in st.session_state:
        st.session_state.selected_token = None

    if st.session_state.selected_token is None:
        st.error("Please select a token to verify")
        return

    st.write("Selected token:")
    st.write(st.session_state.selected_token)
    (msg_offset, token_offset, next_token) = st.session_state.selected_token

    # TODO: make this match the real history
    messages = st.session_state.messages[:msg_offset]
    messages.append(
        {
            "role": st.session_state.messages[msg_offset]["role"],
            "content": st.session_state.messages[msg_offset]["content"][:token_offset],
        }
    )
    st.write("Chat history:")
    st.markdown(
        "```\n" + "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\n```"
    )
    st.write("Next token:")
    st.markdown("`" + next_token + "`")

    st.header("Check with local computation")
    if st.button("Start"):
        with st.status("running..."):
            response = requests.post(
                "http://localhost:19001/forward",
                json=messages,
            )
            token = response.json()
        st.write("Finished:")

        # TODO: replace with actual metrics
        st.metric("Time used (s):", 1.0)
        st.write(f"The token is: `{token}`")

    st.header("Generate inference trace")
    if st.button("Generate"):
        if "traces" not in st.session_state:
            with st.status("running..."):
                begin = time.time()
                next_token, tensors, probs, output2input = request_traces(messages)
                probs_chart_data = pd.DataFrame(
                    {
                        "token": [
                            st.session_state.tokenizer.decode([p[0]]) for p in probs
                        ],
                        "probability": [p[1] for p in probs],
                    }
                )
                c = (
                    alt.Chart(probs_chart_data)
                    .mark_bar()
                    .encode(
                        x="token",
                        y="probability",
                        color=alt.value("steelblue"),
                    )
                )
                st.altair_chart(c, use_container_width=True)
                generation_time = time.time() - begin
                st.session_state.dumped_traces = pickle.dumps((tensors, output2input))
            st.metric("Generation time used (s):", generation_time)

        st.download_button(
            f"Download trace (approximately {len(st.session_state.dumped_traces) // 1024 // 1024} MB)",
            st.session_state.dumped_traces,
            "trace.pkl",
            "application/octet-stream",
            type="primary",
        )

    st.header("Check the trace")
    if uploaded_file := st.file_uploader("Choose a trace file"):
        tensors, output2input = pickle.loads(uploaded_file.getvalue())
        traces = {OpId.from_str(k): v for k, v in tensors.items()}
        if (
            "diffs" not in st.session_state
        ):  # need to figure out a away for submitting different traces
            with st.spinner("Checking trace using deterministic verification..."):
                begin = time.time()
                metadata = {
                    "messages": messages,
                }
                response = requests.post(
                    "http://localhost:19001/check",
                    data=dumps(tensors, metadata),
                )
                st.session_state.diffs = {
                    OpId.from_str(k): v for k, v in response.json().items()
                }
                checking_time = time.time() - begin
                st.metric("Checking time used (s):", checking_time)

        diffs = st.session_state.diffs
        chart_data = pd.DataFrame(
            {
                "op_id": [str(op_id) for op_id in diffs.keys()],
                "diff": [v for v in diffs.values()],
            }
        )
        c = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x="op_id",
                y="diff",
                color=alt.value("steelblue"),
            )
        )
        st.altair_chart(c)
        threshold = st.slider("Threshold for verification", 0.0, 0.1, 0.0, 0.001)

        if all(v <= threshold for v in diffs.values()):
            st.success(f"The trace is correct with threshold {threshold}")
        else:
            st.error("The trace is incorrect")
            if "fraud_proof" not in st.session_state:
                for output_id, diff in diffs.items():
                    if diff > threshold:
                        input_ids = output2input[output_id]
                        all_ids = input_ids + [output_id]
                        partial_traces = {str(id): traces[id] for id in all_ids}
                        output_id = str(output_id)
                        st.session_state.fraud_proof = (
                            output_id,
                            threshold,
                            partial_traces,
                        )
            st.download_button(
                "Fraud proof",
                pickle.dumps(st.session_state.fraud_proof),
                "proof.pkl",
                "application/octet-stream",
                type="primary",
            )

    st.header("Verify the proof")
    if uploaded_file := st.file_uploader("Choose a proof file"):
        with st.spinner("Verifying the proof..."):
            begin = time.time()
            output_id, threshold, tensors = pickle.loads(uploaded_file.getvalue())
            response = requests.post(
                "http://localhost:19001/verify",
                data=dumps(
                    tensors,
                    {"op_id": output_id, "threshold": threshold, "messages": messages},
                ),
            )
            if not response.json():
                st.success("The proof is correct")
            else:
                st.error("The proof is incorrect")
            st.metric("Verification time used (s):", time.time() - begin)


page_verify = st.Page(
    f_page_verify, title="Inference Result Verification", icon=":material/favorite:"
)

pg = st.navigation([page_chat, page_verify])
pg.run()
