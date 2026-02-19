from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import os
from hmm import baum_welch

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        obs_input = request.form["observations"]
        N = int(request.form["states"])

        # Normalize and tokenize
        obs_tokens = obs_input.lower().split()

        # Create observation vocabulary
        unique_obs = list(set(obs_tokens))
        obs_map = {obs: i for i, obs in enumerate(unique_obs)}

        obs_seq = np.array([obs_map[o] for o in obs_tokens])
        M = len(unique_obs)

        pi, A, B, likelihoods = baum_welch(obs_seq, N, M, iterations=20)

        # Plot likelihood
        plt.figure()
        plt.plot(likelihoods)
        plt.xlabel("Iteration")
        plt.ylabel("P(O | λ)")
        plt.title("Likelihood over Iterations")
        plt.tight_layout()
        plt.savefig("static/likelihood.png")
        plt.close()

        result = {
            "pi": pi,
            "A": A,
            "B": B,
            "likelihood": likelihoods[-1],
            "obs_tokens": obs_tokens,
            "states": [f"S{i}" for i in range(N)],
            "vocab": unique_obs
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
