from flask import Flask, render_template, request
import copy
from hmm import (
    states,
    observations,
    start_prob,
    transition_prob,
    emission_prob,
    baum_welch,
    viterbi_algorithm
)


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        obs_input = request.form["observations"]
        # Normalize input: remove extra spaces + make case-insensitive
        obs_sequence = [
            word.strip().capitalize()
            for word in obs_input.split()
        ]

        # Validate observations
        invalid = [obs for obs in obs_sequence if obs not in observations]

        if invalid:
            return render_template(
                "index.html",
                result=None,
                error=f"Invalid observation(s): {', '.join(invalid)}",
                states=states,
                observations=observations
            )



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
        # Run Viterbi on trained model
        viterbi_prob, best_path = viterbi_algorithm(
            obs_sequence,
            states,
            new_start,
            new_trans,
            new_emit
        )


        # Count observation frequency
        obs_counts = {}
        for obs in obs_sequence:
            obs_counts[obs] = obs_counts.get(obs, 0) + 1

        explanation_details = []

        # Start probability comparison
        for state in states:
            old_val = original_start[state]
            new_val = new_start[state]
            diff = new_val - old_val

            explanation_details.append(
                f"Start Probability for {state}: "
                f"{old_val:.4f} → {new_val:.4f} "
                f"(Change: {diff:.4f})"
            )

        # Emission comparison
        for state in states:
            for obs in observations:
                old_val = original_emit[state][obs]
                new_val = new_emit[state][obs]
                diff = new_val - old_val

                explanation_details.append(
                    f"Emission P({obs}|{state}): "
                    f"{old_val:.4f} → {new_val:.4f} "
                    f"(Change: {diff:.4f})"
                )

        # Add frequency-based reasoning
        for obs, count in obs_counts.items():
            explanation_details.append(
                f"Observation '{obs}' appeared {count} times in the sequence."
            )

        # Generate simple explanation
        explanation = []

        for state in states:
            if new_start[state] > original_start[state]:
                explanation.append(
                    f"The probability of starting in {state} increased because the observations align more with this state."
                )
            else:
                explanation.append(
                    f"The probability of starting in {state} decreased as the data supports other states more strongly."
                )

        for state in states:
            for obs in observations:
                if new_emit[state][obs] > original_emit[state][obs]:
                    explanation.append(
                        f"The model now associates '{obs}' more strongly with state '{state}'."
                    )

        result = {
            "original_start": original_start,
            "original_trans": original_trans,
            "original_emit": original_emit,
            "new_start": new_start,
            "new_trans": new_trans,
            "new_emit": new_emit,
            "best_path": best_path,
            "viterbi_prob": viterbi_prob,
            "obs_sequence": obs_sequence,
            "explanation_details": explanation_details
        }




    return render_template("index.html", result=result, states=states, observations=observations)


if __name__ == "__main__":
    app.run(debug=True)
