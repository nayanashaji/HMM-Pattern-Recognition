# Hidden Markov Model – Baum-Welch Implementation

**Name:** Nayana Shaji Mekkunnel

**Roll No:** 50

**University Registration No:** TCR24CS051

S4 CSE


## Overview

This project implements a **Hidden Markov Model (HMM)** with the **Baum–Welch algorithm** (Expectation–Maximization) from scratch in Python.

The application:

* Accepts an observed sequence
* Accepts number of hidden states
* Learns model parameters using Baum–Welch
* Computes likelihood ( P(O \mid \lambda) )
* Visualizes likelihood over iterations
* Generates state transition diagrams

The project is built using:

* Flask (Web interface)
* NumPy (Matrix computations)
* Matplotlib (Graphs)
* NetworkX (State transition diagrams)

---

## Features

### 1. Inputs

* Observation sequence (space-separated)
* Number of hidden states (N)

### 2. Outputs

* Initial distribution ( \pi )
* Transition matrix ( A )
* Emission matrix ( B )
* Final likelihood ( P(O \mid \lambda) )
* Likelihood graph over iterations
* State transition diagram

### 3. Algorithms Implemented

* Forward algorithm
* Backward algorithm
* Gamma computation
* Xi computation
* Baum–Welch parameter re-estimation
* Log-likelihood tracking

---

## Mathematical Formulation

The Baum–Welch algorithm iteratively updates:

### Initial Distribution

$$
\pi_i = \gamma_1(i)
$$

### Transition Matrix

$$
a_{ij} =
\frac{\sum_{t=1}^{T-1} \xi_t(i,j)}
{\sum_{t=1}^{T-1} \gamma_t(i)}
$$

### Emission Matrix

$$
b_i(k) =
\frac{\sum_{t: O_t = k} \gamma_t(i)}
{\sum_{t=1}^{T} \gamma_t(i)}
$$

The likelihood is computed using the Forward algorithm:

$$
P(O \mid \lambda)
$$

## Project Structure

```
HMM-Project/
│
├── app.py
├── general_hmm.py
├── requirements.txt
├── Procfile (if deployed)
├── static/
│   ├── likelihood.png
│   ├── state_diagram.png
│   └── style.css
└── templates/
    └── index.html
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/nayanashaji/HMM-Pattern-Recognition.git
cd HMM-Pattern-Recognition
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install flask numpy matplotlib networkx
```

---

## Running Locally

```
python app.py
```

Then open:

```
http://127.0.0.1:5000/
```

---

## Deployment

This project has been deployed on https://hmm-pattern-recognition.onrender.com


## Example Usage

Input:

```
walk shop clean walk shop
```

Hidden States:

```
3
```

Output:

* Learned π
* Learned A
* Learned B
* Log-likelihood convergence graph
* State transition diagram

---

## Learning Behavior

The log-likelihood increases across iterations, demonstrating:

* Expectation step (state probability estimation)
* Maximization step (parameter update)
* Convergence toward a local maximum

---

## Notes

* Initialization is random (Dirichlet-based normalization).
* Different runs may produce different models (local maxima).
* Log-likelihood is used for numerical stability.
* Transition diagrams are generated using NetworkX.

---

## Author

Implemented as part of coursework on Hidden Markov Models and the Baum–Welch algorithm.

---

