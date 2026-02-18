from flask import Flask, render_template, request
import copy
from hmm import (
    states,
    observations,
    start_prob,
    transition_prob,
    emission_prob,
    baum_welch
)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        obs_input = request.form["observations"]
        obs_sequence = obs_input.split()

        # Keep original values (deep copy)
        original_start = copy.deepcopy(start_prob)
        original_trans = copy.deepcopy(transition_prob)
        original_emit = copy.deepcopy(emission_prob)

        # Train using Baum-Welch
        new_start, new_trans, new_emit = baum_welch(
            obs_sequence,
            states,
            copy.deepcopy(start_prob),
            copy.deepcopy(transition_prob),
            copy.deepcopy(emission_prob),
            iterations=10
        )

        result = {
            "original_start": original_start,
            "original_trans": original_trans,
            "original_emit": original_emit,
            "new_start": new_start,
            "new_trans": new_trans,
            "new_emit": new_emit
        }

    return render_template("index.html", result=result, states=states, observations=observations)

if __name__ == "__main__":
    app.run(debug=True)
