# NFL Fourth Down Decision Analyzer

An end-to-end machine learning project that predicts the probability an NFL team will go for it on fourth down based on game context.
The model is deployed with an interactive Streamlit web interface that allows users to explore real-time what-if scenarios without any local setup.

**Live App:** https://nflfourthdownanalyzer.streamlit.app/

## Overview

Fourth-down decisions are among the most consequential and debated choices in football. This project uses historical NFL play-by-play data to model real-world fourth-down behavior and estimate the probability a team chooses to go for it versus kicking (punt or field goal).

Users can input a game situation via a scoreboard-style interface and dynamically adjust inputs (yards to go, score differential, clock, timeouts) to see how predicted probabilities change.

### Data Source

NFL Play-by-Play Data (nflfastR)

Source: https://github.com/nflverse/nflfastR-data

The dataset includes detailed play-level information for multiple NFL seasons.
Only fourth-down plays were used, excluding kneels and no-plays.

## Features Used

The model uses the following features:

`ydstogo` — yards needed for a first down

`yardline_100` — distance (yards) from the opponent’s end zone
(5 = 5 yards from scoring, 100 = own goal line)

`score_differential` — offense score minus defense score

`game_seconds_remaining` — total seconds remaining in the game

`posteam_timeouts_remaining` — offensive timeouts remaining

These features were selected for interpretability and alignment with real coaching decisions.

## Modeling Approach

**Target:**
Binary classification

1 = go for it (run or pass)

0 = kick (punt or field goal)

### Models explored:

Logistic regression (baseline)

Feedforward neural network (final model)

### Final model:

PyTorch neural network

2 hidden layers (64 → 32)

ReLU activations

Softmax output for probabilistic predictions

### Performance

Evaluated on a held-out test set:

Test AUC: ~0.92–0.93

Test Accuracy: ~0.91

True go rate: ~14%

Predicted go rate: closely matched true rate

This indicates strong discrimination while maintaining realistic calibration.

## Interactive Web App

The Streamlit app allows users to:

Enter game context using intuitive scoreboard-style inputs

Adjust inputs using sliders for what-if analysis

View real-time probability estimates

Explore how decisions change with clock, score, and field position

The model runs entirely on the server — users do not need to install Python or download the model.

## Project Structure
├── app.py                  # Streamlit frontend
├── model.py                # Neural network definition
├── data_load.py            # Data loading & preprocessing
├── Models/
│   └── fourth_down_nn_model.pth
│   └── fourth_down_logreg.pkl
├── requirements.txt
└── README.md

## Tech Stack

Python

PyTorch

scikit-learn

pandas / NumPy

Streamlit

GitHub + Streamlit Cloud

## Notes & Limitations

The model predicts historical decision behavior, not necessarily optimal strategy

Does not explicitly account for:

kicker quality

weather

team-specific tendencies

Overtime scenarios are not currently modeled

## Future Work

Add overtime support

Incorporate team- or coach-specific effects

Add model interpretability tools (e.g., SHAP-style explanations)

Compare predicted decisions to analytically optimal choices

Log anonymized usage patterns to study decision sensitivity

## Author

Rahul Govil - LinkedIn: https://www.linkedin.com/in/govrahul

This project was built as an applied machine learning portfolio piece and is intended for educational and analytical use.
