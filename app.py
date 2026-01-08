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

st.title("NFL 4th Down Decision Analyzer")
st.markdown("Enter game context to estimate the probability of going for it on 4th down or use the sliders below for what-if scenarios.")

# Inputs
st.subheader("Game Situation")
left, right = st.columns(2)

with left:
    st.markdown("**Field Position**")
    ydstogo = st.number_input("Yards to Go", 1, 30, 10)
    yardline_100 = st.number_input("Yardline (100 = own goal)", 1, 99, 45)

    st.markdown("**Game Clock**")
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    time_str = st.text_input("Time Remaining in Quarter (MM:SS)", "10:00")
    try:
        minutes, seconds = map(int, time_str.split(":"))
        if not (0 <= minutes <= 15 and 0 <= seconds < 60):
            st.error("Minutes must be 0-15 and seconds 0-59")
            minutes, seconds = 10, 0
    except ValueError:
        st.error("Enter time in MM:SS format, e.g., 10:00")
        minutes, seconds = 10, 0

with right:
    st.markdown("**Game Context**")
    score_diff = st.number_input("Score Differential", -50, 50, 0)
    timeouts = st.number_input("Timeouts Remaining", 0, 3, 3)

seconds_left_in_quarter = minutes * 60 + seconds
game_time_remaining = (4 - quarter) * 900 + seconds_left_in_quarter

# Optional sliders
st.subheader("What-if Analysis")

ydstogo_sim = st.slider("Adjust Yards to Go", 1, 30, ydstogo)
score_diff_sim = st.slider("Adjust Score Differential (Offense - Defense)", -50, 50, score_diff)
timeouts_sim = st.slider("Adjust Timeouts Remaining", 0, 3, timeouts)
yardline_100_sim = st.slider("Adjust Yardline (100 = own goal)", 1, 100, yardline_100)
minutes_sim = st.slider("Adjust Minutes Remaining", 0, 15, minutes)
seconds_sim = st.slider("Adjust Seconds Remaining", 0, 59, seconds)

seconds_left_sim = minutes_sim * 60 + seconds_sim
time_sim = (4 - quarter) * 900 + seconds_left_sim

# Prepare inputs and predict
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