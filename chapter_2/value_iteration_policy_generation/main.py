import numpy as np

SQUARES = 5
STARTING_POSITION = 0
GOAL_SQUARE = SQUARES - 1

ACTION_LEFT = 0
ACTION_RIGHT = 1

NUM_STATES = GOAL_SQUARE
NUM_ACTIONS = 2

REWARD = SQUARES
LAST_VALID_STATE = GOAL_SQUARE - 1

def is_valid_state(state):
   return state >= 0 and state <= LAST_VALID_STATE

"""
Dynamic programming (DP) methods bootstrap [organizować coś przy użyciu minimalnych zasobów, też: uruchomić]
by updating the policy after a single time step. But DP algorithms must have complete knowledge
of the transition probabilities and visit every possible state and action before they
can find an optimal policy.

Edit: Za chwilę wyjaśnił bootstrapping:
A wide range of disciplines use the term bootstrapping to mean the entity can "lift itself up".
Business bootstrap by raising cash without any loans.

In statistics and RL, bootstrapping is a sampling method that uses individual observations to estimate the
statistics of the population.
"""
# theta - stopping threshold, theta > 0,
# it controls how long to keep refining the value estimate
def value_iteration(gamma, theta):
   # transition probabilities: p(s', r | s, a) - prawdopodobieństwo skończenia
   # w stanie s' z nagrodą r mając daną akcję a i stan s
   #
   # lista [action][state][[probability, next_state, reward]] (tyle tablic wewnętrznych ile next_state'ów czyli u mnie po jednej, ale
   # w ogólnym przypadku może być więcej - wtedy probability musi się podzielić między te next_state'y żeby się sumować do 1 (100%))
   transition_probabilities = [[[[1, -1, 0]] for i in range(NUM_STATES)] for j in range(NUM_ACTIONS)]

   # Ustaw następne stany dla wszystkich akcji
   for state in range(NUM_STATES):
      transition_probabilities[ACTION_LEFT][state][0][1] = state - 1
      transition_probabilities[ACTION_RIGHT][state][0][1] = state + 1

   # Nagroda jest przyznana tylko jeśli jestem na przedostatnim kafelku i idę w prawo
   transition_probabilities[ACTION_RIGHT][LAST_VALID_STATE][0][2] = REWARD

   best_actions = [0] * NUM_STATES
   state_value_buffer = [0] * NUM_STATES
   
   while True:
      """
      ** UWAGA **
      to co jest podane w książce nie działa, dopiero na tej stronie: https://core-robotics.gatech.edu/2021/01/19/bootcamp-summer-2020-week-3-value-iteration-and-q-learning/
      jest alternatywna wersja tego algorytmu i tam jest nabla ustawiana za każdym razem na 0 przed pętlą obsługującą wszystkie stany.
      
      W książce tego nie ma - tam nabla jest tylko raz ustawiana przed wszystkimi pętlami, na samym początku. Przez to ona będzie tylko rosnąć
      (bo dalej jest robione nabla = max(nabla, x) więc będzie albo takie same albo większe), więc nigdy nie będzie mniejsza niż theta więc algorytm
      nigdy się nie zakończy. (Chyba że się zakończy już w pierwszej iteracji. Jeśli nie to w drugiej też nie i w kolejnych też nie (bo będzie albo taki
      sam albo większy, a jeśli w pierwszej był większy niż theta to w kolejnych też))
      """
      # nabla to odwrócona delta (w sensie trójkąt) it maintains the current amount of error
      nabla = 0

      for state in range(NUM_STATES):
         expected_returns = [0] * NUM_ACTIONS

         for action in range(NUM_ACTIONS):
            transitions = transition_probabilities[action][state]

            for probability, next_state, reward in transitions:
               expected_returns[action] += probability * (reward + (gamma * state_value_buffer[next_state] if is_valid_state(next_state) else 0))

         best_action = np.argmax(expected_returns)

         best_actions[state] = best_action

         v = state_value_buffer[state]

         state_value_buffer[state] = expected_returns[best_action]

         nabla = max(nabla, abs(v - state_value_buffer[state]))

      if nabla <= theta:
         break
   
   return state_value_buffer, best_actions

state_value_buffer, best_actions = value_iteration(0.9, 3.7)

print(f"Best actions (optimal policy): {best_actions}, V(S)={state_value_buffer}")