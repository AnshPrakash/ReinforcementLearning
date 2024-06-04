from Ex1_env_HB import HikerAndBear
from mushroom_rl.core.serialization import Serializable
import numpy as np
import matplotlib.pyplot as plt

agents_path = "agents/"
env = HikerAndBear()

agent_path = agents_path + "EpsGreedy_SARSALambda.zip"
agent = Serializable.load(agent_path)

MAX_LEN = 100
episodes = []
# 3 trajectories by following agents policy
for i in range(3):
    curr_state = env.reset()
    print(curr_state)
    absorbing = not np.any(env.p[curr_state[0]])
    episode = []
    while not absorbing and len(episode) < MAX_LEN:
        action = agent.draw_action(curr_state)
        next_state, reward, absorbing, _ = env.step(action)
        episode.append( (next_state, action, curr_state) )
        curr_state = next_state
    episodes.append(episode)


def visualise(episodes):
    
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    H,W = 10, 10
    grid = np.ones((H,W), dtype=int)*-1
    directions = [[0, 1], [-1, 0], [0, -1], [1, 0]]
    # for i, c in enumerate(cell_list):
    #   pi[tuple(c)]=np.argmax(Q[i])
    
    for idx in range(3):
        grid[5][5] = 0
        grid[5][0] = 0
        
        curr_state = episodes[idx][0][2]
        x , y =  (curr_state//W).item()  , (curr_state%W).item()
        if curr_state >= 100:
            curr_state = curr_state%100
            x , y =  (curr_state//W).item()  , (curr_state%W).item()
        grid[x][y] = 1
        if episodes[idx][0][0] >= 100:
            grid[5][0] = -1
        else:
            grid[5][5] = -1
        ax[idx].imshow(grid, cmap='jet')
        grid[x][y] = -1
        grid[5][5] = -1
        grid[5][0] = -1

    # draw the policy arrows
    colors = ['w', 'b', 'g']
    for idx, episode in enumerate(episodes):
        for step in range(len(episode)):
            obs = episode[step]
            curr_state = obs[2]
            i , j =  (curr_state//W).item()  , (curr_state%W).item()
            if curr_state >= 100:
                curr_state = curr_state%100
                i , j =  (curr_state//W).item()  , (curr_state%W).item()
            
            action = obs[1].item()
            ax[idx].arrow(j, i, 0.25*directions[action][1], 0.25*directions[action][0], head_width=0.2, head_length=0.1, color=colors[idx])
    plt.show()
    
    

visualise(episodes)
