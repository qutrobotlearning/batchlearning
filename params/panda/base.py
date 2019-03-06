import session
import task

def create_env():
  import env
  return env.ReacherEnv()

SESSION    = session.Session
TASK       = task.Task
SEED       = 0
ENV        = "PandaReacher"
CREATE_ENV = create_env
UPDATES    = 200000
DATA_IN    = "live"
DATA_OUT   = None

