import session
import task
import gym

SESSION    = session.Session
TASK       = task.Task
SEED       = 0
ENV        = "HalfCheetah-v1"
CREATE_ENV = lambda: gym.make(ENV)
UPDATES    = 10000 # TODO: CHANGE BACK TO 1M !!!
DATA_IN    = "live"
DATA_OUT   = None # name this dynamically

