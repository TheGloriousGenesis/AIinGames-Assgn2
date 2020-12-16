################ Environment ################

import contextlib
from itertools import compress

import numpy as np


# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)

        
class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        
        return next_state, reward

        
class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        
        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            # todo is this ok? as to not start in absorbing state
            self.pi = np.full(n_states, 1./(n_states - 1))
            # make last
            self.pi[-1] = 0
        
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        
        return self.state
        
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        
        self.state, reward = self.draw(self.state, action)
        
        return self.state, reward, done
    
    def render(self, policy=None, value=None):
        raise NotImplementedError()

        
class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        
        self.slip = slip

        # additional state added for absorbing state
        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=None)
        self.P = {s: {a: [] for a in range(n_actions)} for s in range(n_states)}

        nrow = ncol = len(lake)

        # converts from matrix representation of state to linear array representation. index start from 0
        def to_s(row, col):
            return row*ncol + col

        # describes row/ column you will end up on if action a is taken. calculates using min/max to ensure negative values
        # or off grid values are never chosen for row or column
        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            # calculates new row, col one will be on choosing said action
            newrow, newcol = inc(row, col, action)
            # retrieves the state this corresponds to a flattened map representation of states
            newstate = to_s(newrow, newcol)
            # retrieves char representation of state (if goal/hole/ice)
            newletter = self.lake[newrow, newcol]
            # checks if you have reached goal state or if you have dropped into a hole as well
            # as the reward for that state
            done = newletter == '$' or newletter == '#'
            if newletter == '$':
                reward = 1.0
            else:
                reward = 0.0
            return newstate, reward, done

        # for each element in the matrix of states
        for row in range(nrow):
            for col in range(ncol):
                # take the linear representation of the state. starts from 0 to n_states - 1
                s = to_s(row, col)
                # for each action
                for a in range(4):

                    li = self.P[s][a]
                    letter = self.lake[row, col]
                    # if goal or hole populate value with the following:
                    # [probability, newstate, reward, done]
                    if letter == '$':
                        li.append((1.0, s, 1.0, True))
                    elif letter == '#':
                        li.append((1.0, n_states - 1, 0, True))
                    else:
                        # if slippery, check all actions permissiable from state and find outcome of them
                        # for each action there is a list of possible outcomes if not goal/hole
                        # 0.1 of the time pick random direction (out of three 0.03333)
                        # 0.9 of time pick assigned direction
                        for bi in range(4):
                            if bi == a:
                                li.append((
                                    1 - slip,
                                    *update_probability_matrix(row, col, bi)
                                ))
                            else:
                                li.append((
                                    slip / 3,
                                    *update_probability_matrix(row, col, bi)
                                ))
        for a in range(4):
            absorb_state = self.P[len(self.P) - 1][a]
            absorb_state.append((1.0, n_states - 1, 0, True))

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
        
    def p(self, next_state, state, action):
        # TODO:
        if state == self.n_states - 1 == next_state:
            return 1.0

        res = self.P[state][action]
        possible_travel_to_states = [a[1] for a in res]
        if next_state in possible_travel_to_states:
            y = [i == next_state for i in possible_travel_to_states]
            z = list(compress(res, y))
            return sum([i[0] for i in z])
        return 0.0
    
    def r(self, next_state, state, action):
        # TODO:
        # returns the state that is at the last square on grid
        if state == self.n_states - 2:
            return 1.0
        else:
            return 0.0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            ## todo: why this?
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['<', '_', '>', '^']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

def play(env):
    # LEFT = 0
    # DOWN = 1
    # RIGHT = 2
    # UP = 3
    actions = ['a', 's', 'd', 'w']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))


################ Model-based algorithms ################


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    """
    :param policy:          Current policy under question
    :param gamma:           Discount factor
    :param theta:           Minimumal difference allowed value
    :param max_iterations:  Maximum number of iterations when estimating value function
    :return:                Vector of length policy representing the value function.

    When there can be multiple actions for a given state (non deterministic) then iterate over all actions
    then use below instead.

    sum_v_for_all_actions = 0
        for a in range(self.n_actions):
            for ns in range(self.n_states):
                sum_v_for_all_actions += policy[s] * self.p(ns, s, policy[s]) *
                (self.r(ns, s, policy[s]) + gamma * value[ns])
    """
    value = np.zeros(env.n_states, dtype=np.float)
    count = 0
    while True and count < max_iterations:
        delta = 0
        count = count + 1

        for s in range(env.n_states):
            tmp = value[s]
            value[s] = sum([env.p(ns, s, policy[s]) * (env.r(ns, s, policy[s]) + (gamma * value[ns]))
                            for ns in range(env.n_states)])
            delta = max(delta, abs(tmp - value[s]))
        if delta < theta:
            break
    return value


def policy_improvement(env, policy, value, gamma):
    """
    :param policy:  Current policy under evaluation
    :param value:   Value function for current policy
    :param gamma:   Discount factor
    :return:        Returns an improved policy (array of all possible positions in the game)
    """
    improved_policy = np.zeros(env.n_states, dtype=int)
    
    # TODO:
    # todo: this does not take in policy how are we suppose to do improvement on it if not here?
    # Have edited this to take in the policy under improvement evalution. Is this correct?
    # you can actually find improvement without using policy, just need to calculate if improved policy is
    # is improved outside in policy iteration. the comparison does not need to happen in this method (abstraction of method)
    stable = True
    for s in range(env.n_states):
        chosen_action = policy[s]
        best_action = np.argmax([sum([env.p(ns, s, a) * (env.r(ns, s, a) + gamma * value[ns])
                                      for ns in range(env.n_states)]) for a in range(env.n_actions)])
        if chosen_action != best_action:
            stable = False
        improved_policy[s] = best_action
    return improved_policy, stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    value = np.zeros(env.n_states, dtype=np.float)

    best_policy_found = False
    count = 0

    while count <= max_iterations and not best_policy_found:
        count = count + 1
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy, best_policy_found = policy_improvement(env, policy, value, gamma)

    return policy, value, count


def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)

    count = 0

    while count < max_iterations:
        count = count + 1
        delta = 0.

        for s in range(env.n_states):
            tmp = value[s]
            value[s] = max([sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + (gamma * value[next_s]))
                                 for next_s in range(env.n_states)]) for a in range(env.n_actions)])
            delta = max(delta, np.abs(tmp - value[s]))
        if delta < theta:
            break

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        action_index = np.argmax([sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + (gamma * value[next_s]))
                                       for next_s in range(env.n_states)]) for a in range(env.n_actions)])
        policy[s] = action_index

    return policy, value, count

################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    done = 0

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        while not done:
            a = epsilon_greedy(q, s, epsilon[i], random_state)
            next_s, r, done = env.step(a)
            q[s, a] = q[s, a] + eta[i] * (r + gamma * max(q[next_s, :]) - q[s, a])
            s = next_s

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def epsilon_greedy(q, s, epsilon, random_state):
    if random_state.uniform(0, 1) < epsilon:
        # select a random action
        action = np.random.randint(0, 4)
    else :
        # select a best action
        action = np.argmax(q[s, :])
    return action
################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO:
    
    return theta

def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    q = np.zeros(env.n_actions)

    for i in range(max_episodes):
        features = env.reset()
        done = 0
        s = random_state.randint(0, env.n_states)
        #
        # TODO:
        for a in range(env.n_actions):
            index = np.ravel_multi_index((s, a), (env.n_states, env.n_actions))
            q[a] = np.sum(theta[index] * features[a, index])

        while not done:
            a = epsilon_greedy_feature(q, epsilon[i])
            next_s, r, done = env.step(a)
            delta = r - q[a]

            for a in range(env.n_actions):
                index = np.ravel_multi_index((next_s, a), (env.n_states, env.n_actions))
                q[a] = np.sum(theta[index] * features[a, index])

            next_a = epsilon_greedy_feature(q, epsilon[i])
            delta = delta + gamma * q[next_a]
            theta = theta + eta[i] * delta * features
            s = next_s
    return theta

def epsilon_greedy_feature(q, epsilon, random_state):
    if random_state.random.uniform(0, 1) < epsilon:
        # select a random action
        action = np.random.randint(0, 4)
    else :
        # select a best action
        action = np.argmax(q)
    return action

################ Main function ################

def main():
    seed = 0
    
    # # Small lake
    # lake =   [['&', '.', '.', '.'],
    #           ['.', '#', '.', '#'],
    #           ['.', '.', '.', '#'],
    #           ['#', '.', '.', '$']]

    # Big lake
    lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', '.', '#', '.'],
            ['.', '#', '.', '.', '#', '.', '#', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '$']]


    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 1000
    
    print('')
    
    print('## Policy iteration')
    policy, value, count = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    print('Count: ' + str(count))

    print('')
    
    print('## Value iteration')
    policy, value, count = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    print('Count: ' + str(count))
    
    print('')
    
    # print('# Model-free algorithms')
    # max_episodes = 2000
    # eta = 0.5
    # epsilon = 0.5
    #
    # print('')
    #
    # print('## Sarsa')
    # policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # print('## Q-learning')
    # policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    # env.render(policy, value)
    #
    # print('')
    #
    # linear_env = LinearWrapper(env)
    #
    # print('## Linear Sarsa')
    #
    # parameters = linear_sarsa(linear_env, max_episodes, eta,
    #                           gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)
    #
    # print('')
    #
    # print('## Linear Q-learning')
    #
    # parameters = linear_q_learning(linear_env, max_episodes, eta,
    #                                gamma, epsilon, seed=seed)
    # policy, value = linear_env.decode_policy(parameters)
    # linear_env.render(policy, value)


if __name__ == '__main__':
    seed = 0
    #
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    # # Small lake
    # lake = [['&', '.', '.', '.'],
    #           ['.', '#', '.', '#'],
    #           ['.', '.', '.', '#'],
    #           ['#', '.', '.', '$']]
    #
    # Big lake
    # lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '.', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '.', '.', '.', '.', '#', '.', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '.'],
    #         ['.', '#', '#', '.', '.', '.', '#', '.'],
    #         ['.', '#', '.', '.', '#', '.', '#', '.'],
    #         ['.', '.', '.', '#', '.', '.', '.', '$']]

    # play(FrozenLake(lake, slip=0.1, max_steps=(len(lake)**2), seed=seed))
    main()


