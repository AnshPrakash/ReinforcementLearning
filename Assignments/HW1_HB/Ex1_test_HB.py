from Ex1_env_HB import HikerAndBear

env = HikerAndBear()
keys_to_action = {"d": 0, "w": 1, "a": 2, "s": 3}

from pynput import keyboard


def on_press(key):
    if hasattr(key, "char"):
        if key.char not in keys_to_action:
            return
        action = keys_to_action[key.char]
    else:
        return
    state, reward, done, info = env.step([action])
    env.render()
    print(
        "state:{}, action: {}, reward: {}, done: {}".format(state, action, reward, done)
    )
    if done:
        print("Episode finished")
        env.reset()


with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
