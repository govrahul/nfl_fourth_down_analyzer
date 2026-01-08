import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from modeling_nn import FourthDownNN

# Load pre-trained model and cache
@st.cache_resource
def load_model():
    model = FourthDownNN(input_size=5)
    model.load_state_dict(torch.load('Models/fourth_down_nn_model.pth', map_location="cpu"))
    model.eval()
    return model

model = load_model()

def set_input_mode():
    st.session_state.input_mode = "input"

def set_slider_mode():
    st.session_state.input_mode = "slider"

st.title("NFL 4th Down Decision Analyzer")
st.markdown("Enter game context to estimate the probability of going for it on 4th down or use the sliders below for what-if scenarios.")

# Inputs
tab_input, tab_slider = st.tabs(["Game Situation", "What-if Analysis"])
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "input"

with tab_input:
    st.subheader("Game Situation")
    left, right = st.columns(2)

    with left:
        st.markdown("**Field Position**")
        ydstogo = st.number_input("Yards to Go", 1, 30, 10, on_change=set_input_mode)
        yardline_100 = st.number_input("Yardline (100 = own goal)", 1, 99, 45, on_change=set_input_mode)

        st.markdown("**Game Clock**")
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], key="quarter", on_change=set_input_mode)
        time_str = st.text_input("Time Remaining in Quarter (MM:SS)", "10:00", on_change=set_input_mode)
        try:
            minutes, seconds = map(int, time_str.split(":"))
            if not (0 <= minutes <= 15 and 0 <= seconds < 60):
                st.error("Minutes must be 0-15 and seconds 0-59")
                minutes, seconds = 10, 0
            if (minutes == 15 and seconds > 0):
                st.error("At most 15:00 allowed")
                minutes, seconds = 10, 0
        except ValueError:
            st.error("Enter time in MM:SS format, e.g., 10:00")
            minutes, seconds = 10, 0

    with right:
        st.markdown("**Game Context**")
        score_diff = st.number_input("Score Differential", -50, 50, 0, on_change=set_input_mode)
        timeouts = st.number_input("Timeouts Remaining", 0, 3, 3, on_change=set_input_mode)

seconds_left_in_quarter = minutes * 60 + seconds
game_time_remaining = (4 - quarter) * 900 + seconds_left_in_quarter

# Optional sliders
with tab_slider:
    st.subheader("What-if Analysis")

    quarter_sim = st.selectbox("Quarter", [1, 2, 3, 4], key="quarter_sim", on_change=set_slider_mode)
    ydstogo_sim = st.slider("Adjust Yards to Go", 1, 30, ydstogo, on_change=set_slider_mode)
    score_diff_sim = st.slider("Adjust Score Differential (Offense - Defense)", -50, 50, score_diff, on_change=set_slider_mode)
    timeouts_sim = st.slider("Adjust Timeouts Remaining", 0, 3, timeouts, on_change=set_slider_mode)
    yardline_100_sim = st.slider("Adjust Yardline (100 = own goal)", 1, 100, yardline_100, on_change=set_slider_mode)
    minutes_sim = st.slider("Adjust Minutes Remaining", 0, 15, minutes, on_change=set_slider_mode)
    seconds_sim = st.slider("Adjust Seconds Remaining", 0, 59, seconds, on_change=set_slider_mode)

seconds_left_sim = minutes_sim * 60 + seconds_sim
time_sim = (4 - quarter_sim) * 900 + seconds_left_sim

# Prepare inputs and predict
if st.session_state.input_mode == "input":
    features = np.array([
        ydstogo,
        score_diff,
        game_time_remaining,
        timeouts,
        yardline_100
    ], dtype=np.float32)

else:  # slider mode
    features = np.array([
        ydstogo_sim,
        score_diff_sim,
        time_sim,
        timeouts_sim,
        yardline_100_sim
    ], dtype=np.float32)

x = torch.tensor(features).unsqueeze(0)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    go_prob = probs[0, 1].item()

# Output
st.subheader("Prediction Result")
st.metric(label="Probability of Going for It on 4th Down", value=f"{go_prob:.1%}")
st.progress(go_prob)
if go_prob >= 0.5:
    st.success("Model leans toward **going for it**.")
else:
    st.warning("Model leans toward **kicking** (punt or FG).")

st.caption("Model trained on historical NFL fourth-down decisions.")
st.caption("Developed by Rahul Govil - GitHub: https://github.com/govrahul/nfl_fourth_down_analyzer")