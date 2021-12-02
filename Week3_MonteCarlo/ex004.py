import gym
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

env = gym.make('Blackjack-v0')
num_timesteps = 100

def policy(state):
    return 0 if state[0] > 19 else 1

def generate_episode(policy):
    episode = []
    state = env.reset()
    for t in range(num_timesteps):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode

total_return = defaultdict(float)
N = defaultdict(int)

num_iterations = 500000
for i in range(num_iterations):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        R = (sum(rewards[t:]))
        total_return[state] = total_return[state] + R
        N[state] = N[state] + 1

total_return = pd.DataFrame(total_return.items(), columns=['state', 'total_return'])
N = pd.DataFrame(N.items(), columns=['state', 'N'])
df = pd.merge(total_return, N, on='state')
df['value'] = df['total_return'] / df['N']
print(df.head(10))

print(df[df['state']==(21,9,False)]['value'])