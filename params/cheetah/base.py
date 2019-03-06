import session
import task
import gym

SESSION    = session.Session
TASK       = task.Task
SEED       = 0
ENV        = "HalfCheetah-v1"
CREATE_ENV = lambda: gym.make(ENV)
UPDATES    = 1000000
DATA_IN    = "live"
DATA_OUT   = None # name this dynamically

