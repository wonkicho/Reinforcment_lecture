from bandit import BanditTwoArmedHighLowFixed
import numpy as np
env = BanditTwoArmedHighLowFixed()

count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)

def UCB(i):
    ucb = np.zeros(2)
    if i < 2:
        return i
    else:
        for arm in range(2):
            ucb[arm] = Q[arm] + np.sqrt((2*np.log(sum(count))) / count[arm])
    
    return np.argmax(ucb)

num_rounds = 100
for i in range(num_rounds):
    arm = UCB(i)    
    next_state, reward, done, info = env.step(arm)
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

print(Q)
print('The optimal arm is arm {}'.format(np.argmax(Q)+1))

