# from http://en.wikipedia.org/wiki/Hidden_Markov_model
import random
import math
import itertools
import pdb

def is_approx_1(val):
  return 0.99 <= val and val <= 1.01

# float params
def is_approx_equal(a ,b):
  assert b != 0
  return is_approx_1(a / b)

def draw_uniform(event_to_prob):
  rand = random.uniform(0, 1)
  prob_sum = 0
  assert(is_approx_1(sum(prob for (event, prob) in event_to_prob.items())))
  for (event, prob) in event_to_prob.items():
    prob_sum += prob
    if rand < prob_sum:
      return event
    last_event = event
  return last_event

def test_draw_uniform():
  buckets = {}
  for i in range(0, 100):
    state = draw_uniform(start_probability)
    if not state in buckets:
      buckets[state] = 0
    buckets[state] += 1
  print(buckets)    

# convert list of logprobs (not neglogprobs) to probs, then return sum
def sum_of_probs_from_logprobs(logprobs):
  probs = (math.exp(logprob) for logprob in logprobs)
  return sum(probs)

# discrete Hmm, with discrete set of observations
class Hmm:
  def __init__(self, states, observations, start_probability, transition_probability, emission_probability):
    self.states = states                                  # list of state names (e.g. strings), these are hidden
    self.observations = observations                      # emitted by the states
    self.start_probability = start_probability            # state -> prob
    self.transition_probability = transition_probability  # state -> arcs
    self.emission_probability = emission_probability      # state -> observation probs
    assert self.is_valid()

  # checks invariants
  def is_valid(self):
    """ verify probabilities add up, and mappings are valid """
    stateSet = set(self.states)
    if len(stateSet) != len(self.states):
      return False # dups

    observationSet = set(self.observations)
    if len(observationSet) != len(self.observations):
      return False # dups

    if not set(self.start_probability.keys()) == stateSet:
      return False
    if not is_approx_1(sum(self.start_probability.values())):
      return False

    if not set(self.transition_probability.keys()) == stateSet:
      return False
    for transition_arcs in self.transition_probability.values():
      if not set(transition_arcs.keys()) == stateSet:
        return False
      if not is_approx_1(sum(transition_arcs.values())):
        return False

    if not set(self.emission_probability.keys()) == stateSet:
      return False
    for distribution in self.emission_probability.values():
      if not set(distribution.keys()) == observationSet:
        return False
      if not is_approx_1(sum(distribution.values())):
        return False

    return True

def create_example_hmm():

  states = ['Rainy', 'Sunny']

  observations = ['walk', 'shop', 'clean']

  start_probability = {'Rainy': 0.7, 'Sunny': 0.3}
 
  transition_probability = {
    'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
    'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
  }
 
  emission_probability = { # aka observation_probability
    'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
  }

  return Hmm(states, observations, start_probability, transition_probability, emission_probability)

# returns tuples (state, observation)
def generate_observations(hmm, length):
  tuples = []
  state = draw_uniform(hmm.start_probability)
  for i in range(0, length):
    observation = draw_uniform(hmm.emission_probability[state])
    tuples.append( (state, observation) )
    #print("state {} emits {}".format(state, observation))
    state = draw_uniform(hmm.transition_probability[state])
  return tuples

def test_generate_observations():
  hmm = create_example_hmm()
  generate_observations(hmm, 3)

test_generate_observations()

def score_observation_for_state_seq(hmm, observation_seq, state_seq):
  assert len(state_seq) == len(observation_seq)
  initial_state = state_seq[0]
  logProb = 0
  for i in range(0, len(state_seq)):
    observation = observation_seq[i]
    prevState = state_seq[i-1] if i>0 else None
    state = state_seq[i]
    logProb += math.log(hmm.emission_probability[state][observation])
    logProb += math.log(hmm.start_probability[state]) if prevState == None else math.log(hmm.transition_probability[prevState][state])
  return logProb

# simplest algorithm but with exponential complexity (so only feasible for toy use cases),
# but useful for verifying fwd-backward implementation correctness on such toy cases.
def naive_scoring(hmm, observation_seq):
  logprobs = [] # individual (log)probs for each state seq
  for state_seq in itertools.product(hmm.states, repeat=len(observation_seq)):   # exponential!
    logprobs.append(score_observation_for_state_seq(hmm, observation_seq, state_seq))
  prob_sum = sum_of_probs_from_logprobs(logprobs)
  return prob_sum

# from http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
# I removed end_st aka end_state though (setting a[k][end_st] = 1)
# and split this into explicit fwd and bwd algorithms.
def forward_algorithm(hmm, observation_seq):

    x = observation_seq
    states = hmm.states
    a_0 = hmm.start_probability
    a = hmm.transition_probability
    e = hmm.emission_probability

    L = len(x)
 
    fwd = []
    f_prev = {}
    # forward part of the algorithm
    for i, x_i in enumerate(x):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = a_0[st]
            else:
                prev_f_sum = sum(f_prev[k]*a[k][st] for k in states)
 
            f_curr[st] = e[st][x_i] * prev_f_sum
 
        fwd.append(f_curr)
        f_prev = f_curr

    return fwd # aka alphas

def backward_algorithm(hmm, observation_seq):

    x = observation_seq
    states = hmm.states
    a_0 = hmm.start_probability
    a = hmm.transition_probability
    e = hmm.emission_probability

    L = len(x)
 
    bkw = []
    b_prev = {}
    # backward part of the algorithm
    for i, x_i_plus in enumerate(reversed(x[1:]+[None])):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = 1
            else:
                b_curr[st] = sum(a[st][l]*e[l][x_i_plus]*b_prev[l] for l in states)
 
        bkw.insert(0,b_curr)
        b_prev = b_curr

    return bkw # aka betas

# more efficient variant of naive_scoring() based on forward algorithm
def scoring(hmm, observation_seq):
    fwd = forward_algorithm (hmm, observation_seq)
    last_alphas = fwd[-1]
    p_fwd = sum(last_alphas[k] for k in hmm.states)
    return p_fwd

# returns tuple alphas (from forward algo), betas (from backward algo), posteriors
def forward_backward_algorithm(hmm, observation_seq):
    
    fwd = forward_algorithm (hmm, observation_seq)
    bkw = backward_algorithm(hmm, observation_seq)

    # sanity check P_hmm(observation_seq)
    last_alphas = fwd[-1]
    last_betas  = bkw[0]   # 'last' as in computed last
    p_fwd = sum(last_alphas[k] for k in hmm.states)
    p_bkw = sum(hmm.start_probability[state] * hmm.emission_probability[state][observation_seq[0]] * last_betas[state] for state in hmm.states)
    assert is_approx_equal(p_fwd, p_bkw) # assert == could fail easily due to float rounding errors
 
    # merging the two parts
    posterior = []
    for i in range(len(observation_seq)):
        posterior.append({st: fwd[i][st]*bkw[i][st]/p_fwd for st in hmm.states})
 
    return fwd, bkw, posterior

# example for the HMM problem of computing the prob for an observation_seq
def test_scoring(observation_count):

  hmm = create_example_hmm()

  # let the HMM compute observation seqs and compute a histogram over these
  # to estimate the probs (large sample sets should approximate the observation 
  # seq probs computed by forward algo)
  def observation_seq_to_hashable_value(observation_seq):
    return " ".join(observation_seq) # assuming observation names have no spaces in them

  histogram = {}
  histogram_total_count = 10000
  for i in range(histogram_total_count):
    tuples = generate_observations(hmm, observation_count)
    observation_seq = [observation for (state, observation) in tuples]
    hashable = observation_seq_to_hashable_value(observation_seq)
    if not hashable in histogram:
      histogram[hashable] = 0
    histogram[hashable] += 1

  def getHistogramProb(observation_seq):
    hashable = observation_seq_to_hashable_value(observation_seq)
    return histogram[hashable] / histogram_total_count if hashable in histogram else 0

  # compare naive (exponential complexity) and fwd-bwd algo scoring:
  prob_sum = 0
  for observation_seq in itertools.product(hmm.observations, repeat=observation_count):
    prob = naive_scoring(hmm, observation_seq)   # exponential & trivial algo
    prob2 = scoring(hmm, observation_seq)        # efficient dynamic programming algo (based on either forward or backward algorithm)
    assert is_approx_equal(prob, prob2)
    print("observation_seq {} prob={} histogram_prob={}".format(" ".join(observation_seq), prob, getHistogramProb(observation_seq)))
    prob_sum += prob
  print("  => sum(prob)=", prob_sum)

  # generate a single seq randomly and score it with naive algo & fwd-bwd-algo
  tuples = generate_observations(hmm, observation_count)
  observation_seq = [observation for (state, observation) in tuples]
  prob = naive_scoring(hmm, observation_seq)
  print("random observation_seq {} prob={}".format(" ".join(observation_seq), prob))
  
  (fwd, bkw, posterior) = forward_backward_algorithm(hmm, observation_seq)
  print("alphas:", fwd)
  print("betas:", bkw)
  print("posteriors:", posterior)

#XXX
# Don't study this, it just prints a table of the steps.
def print_dptable(V):
    s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
    for y in V[0]:
        s += "%.5s: " % y
        s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
        s += "\n"
    print(s)

def viterbi(hmm, observation_seq):

    assert all(observation in hmm.observations for observation in observation_seq)
    obs = observation_seq
    states = hmm.states
    start_p = hmm.start_probability
    trans_p = hmm.transition_probability
    emit_p = hmm.emission_probability

    V = [{}]
    path = {}
 
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
 
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
 
        for y in states:
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
 
        # Don't need to remember the old paths
        path = newpath

    n = 0           # if only one element is observed max is sought in the initialization values
    if len(obs) != 1:
        n = t
    print_dptable(V)
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])

def test_viterbi(observation_count):
  print("\n\ntest viterbi")
  hmm = create_example_hmm()

  tuples = generate_observations(hmm, observation_count)
  print("generated random observations: ", tuples)

  observation_seq = [observation for (state, observation) in tuples]

  (viterbi_prob, viterbi_path) = viterbi(hmm, observation_seq)
  print("viterbi path: ", viterbi_path, " with prob ", viterbi_prob)

  # find the viterbi path the inefficient (exponential) way:
  def get_all_path_probs():
    for state_seq in itertools.product(hmm.states, repeat=len(observation_seq)):
      logprob = score_observation_for_state_seq(hmm, observation_seq, state_seq)
      prob = math.exp(logprob)
      yield (state_seq, prob)
  (naive_best_path, naive_best_prob) = max(get_all_path_probs(), key = lambda tuple: tuple[1])
  print("naively computed best_path: ", naive_best_path, " with prob", naive_best_prob)
  assert list(viterbi_path) == list(naive_best_path)
  assert is_approx_equal(viterbi_prob, naive_best_prob)

  # compare viterbi to the alphas computed by the forward_algorithm:
  # the alpha probs are large because they are the sum of probs for
  # all pathes ending in a state, not just the best (viterbi) path:
  alphas = forward_algorithm(hmm, observation_seq)
  print("forward algo alphas=", alphas)

  # compare viterbi to the most likely individual states as computed
  # by forward_backward:
  (alphas, betas, posteriors) = forward_backward_algorithm(hmm, observation_seq)
  print('\nobservation actual_state[i] viterbi_path[i] fwd-bwd[i]')
  for i in range(len(observation_seq)):
    max_posterior_state = max(posteriors[i].items(), key = lambda item: item[1])
    doStatesDiffer = max_posterior_state[0] != viterbi_path[i]
    print("  ", observation_seq[i], tuples[i][0], viterbi_path[i], max_posterior_state, "***" if doStatesDiffer else "")
 
test_scoring(3)

test_viterbi(6)

