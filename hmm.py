# Hidden Markov Model - Weather Example

# Hidden States
states = ["Rainy", "Sunny"]

# Observations
observations = ["Walk", "Shop", "Clean"]

# Initial Probabilities (π)
start_prob = {
    "Rainy": 0.6,
    "Sunny": 0.4
}

# Transition Probabilities (A)
transition_prob = {
    "Rainy": {"Rainy": 0.7, "Sunny": 0.3},
    "Sunny": {"Rainy": 0.4, "Sunny": 0.6}
}

# Emission Probabilities (B)
emission_prob = {
    "Rainy": {"Walk": 0.1, "Shop": 0.4, "Clean": 0.5},
    "Sunny": {"Walk": 0.6, "Shop": 0.3, "Clean": 0.1}
}

# Example observation sequence
obs_sequence = ["Walk", "Shop", "Clean",
    "Walk", "Walk", "Shop",
    "Clean", "Walk", "Shop",
    "Clean"]

def forward_algorithm(obs_seq, states, start_prob, transition_prob, emission_prob):
    
    # Initialize alpha dictionary
    alpha = [{}]
    
    # Step 1: Initialization
    for state in states:
        alpha[0][state] = start_prob[state] * emission_prob[state][obs_seq[0]]
    
    # Step 2: Recursion
    for t in range(1, len(obs_seq)):
        alpha.append({})
        for curr_state in states:
            total = 0
            for prev_state in states:
                total += alpha[t-1][prev_state] * transition_prob[prev_state][curr_state]
            alpha[t][curr_state] = total * emission_prob[curr_state][obs_seq[t]]
    
    # Step 3: Termination
    final_prob = sum(alpha[len(obs_seq)-1][state] for state in states)
    
    return final_prob, alpha

probability, alpha_table = forward_algorithm(
    obs_sequence,
    states,
    start_prob,
    transition_prob,
    emission_prob
)

def viterbi_algorithm(obs_seq, states, start_prob, transition_prob, emission_prob):
    
    # Initialize delta (probabilities) and path
    delta = [{}]
    path = {}

    # Step 1: Initialization
    for state in states:
        delta[0][state] = start_prob[state] * emission_prob[state][obs_seq[0]]
        path[state] = [state]

    # Step 2: Recursion
    for t in range(1, len(obs_seq)):
        delta.append({})
        new_path = {}

        for curr_state in states:
            max_prob = -1
            best_prev_state = None

            for prev_state in states:
                prob = delta[t-1][prev_state] * transition_prob[prev_state][curr_state]
                if prob > max_prob:
                    max_prob = prob
                    best_prev_state = prev_state

            delta[t][curr_state] = max_prob * emission_prob[curr_state][obs_seq[t]]
            new_path[curr_state] = path[best_prev_state] + [curr_state]

        path = new_path

    # Step 3: Termination
    max_final_prob = -1
    best_final_state = None

    for state in states:
        if delta[len(obs_seq)-1][state] > max_final_prob:
            max_final_prob = delta[len(obs_seq)-1][state]
            best_final_state = state

    return max_final_prob, path[best_final_state]

viterbi_prob, best_path = viterbi_algorithm(
    obs_sequence,
    states,
    start_prob,
    transition_prob,
    emission_prob
)

def backward_algorithm(obs_seq, states, transition_prob, emission_prob):
    
    beta = [{} for _ in range(len(obs_seq))]
    
    # Initialization
    for state in states:
        beta[len(obs_seq)-1][state] = 1
    
    # Recursion backwards
    for t in reversed(range(len(obs_seq)-1)):
        for state in states:
            total = 0
            for next_state in states:
                total += (
                    transition_prob[state][next_state]
                    * emission_prob[next_state][obs_seq[t+1]]
                    * beta[t+1][next_state]
                )
            beta[t][state] = total
    
    return beta

def compute_gamma(alpha, beta, states):
    
    gamma = []
    
    for t in range(len(alpha)):
        gamma_t = {}
        denominator = 0
        
        for state in states:
            denominator += alpha[t][state] * beta[t][state]
        
        for state in states:
            gamma_t[state] = (alpha[t][state] * beta[t][state]) / denominator
        
        gamma.append(gamma_t)
    
    return gamma

def compute_xi(obs_seq, alpha, beta, states, transition_prob, emission_prob):
    
    xi = []
    
    for t in range(len(obs_seq)-1):
        xi_t = {}
        denominator = 0
        
        for i in states:
            xi_t[i] = {}
            for j in states:
                value = (
                    alpha[t][i]
                    * transition_prob[i][j]
                    * emission_prob[j][obs_seq[t+1]]
                    * beta[t+1][j]
                )
                xi_t[i][j] = value
                denominator += value
        
        # Normalize
        for i in states:
            for j in states:
                xi_t[i][j] /= denominator
        
        xi.append(xi_t)
    
    return xi

def reestimate_parameters(obs_seq, states, gamma, xi):
    
    # Update initial probabilities (pi)
    new_start_prob = {}
    for state in states:
        new_start_prob[state] = gamma[0][state]

    # Update transition probabilities (A)
    new_transition_prob = {}
    for i in states:
        new_transition_prob[i] = {}
        denominator = sum(gamma[t][i] for t in range(len(gamma)-1))
        
        for j in states:
            numerator = sum(xi[t][i][j] for t in range(len(xi)))
            new_transition_prob[i][j] = numerator / denominator

    # Update emission probabilities (B)
    new_emission_prob = {}
    for state in states:
        new_emission_prob[state] = {}
        denominator = sum(gamma[t][state] for t in range(len(gamma)))
        
        for observation in observations:
            numerator = 0
            for t in range(len(obs_seq)):
                if obs_seq[t] == observation:
                    numerator += gamma[t][state]
            
            new_emission_prob[state][observation] = numerator / denominator

    return new_start_prob, new_transition_prob, new_emission_prob

def baum_welch(obs_seq, states, start_prob, transition_prob, emission_prob, iterations=10):
    
    for _ in range(iterations):
        
        # E-step
        forward_prob, alpha = forward_algorithm(obs_seq, states, start_prob, transition_prob, emission_prob)
        beta = backward_algorithm(obs_seq, states, transition_prob, emission_prob)
        
        gamma = compute_gamma(alpha, beta, states)
        xi = compute_xi(obs_seq, alpha, beta, states, transition_prob, emission_prob)
        
        # M-step
        start_prob, transition_prob, emission_prob = reestimate_parameters(
            obs_seq, states, gamma, xi
        )

    return start_prob, transition_prob, emission_prob

