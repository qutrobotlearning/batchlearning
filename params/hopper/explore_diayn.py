from base  import *
import agent
import task

AGENT     = agent.SoftACAgent
TASK      = task.DIAYNTask
COMMENT   = "{}_{}_{}".format(ENV, AGENT.__name__, TASK.__name__)
LR        = 3e-4
LR_D      = 3e-4
SIGMA_EPS = 1e-5
GAMMA     = 0.99
BATCH     = 256
ALPHA     = 1.0
STEPS     = 1
TAU       = 0.005
RSCALE    = 5.0
L_SIZE    = 50

DATA_OUT  = "{}_live".format(COMMENT)

