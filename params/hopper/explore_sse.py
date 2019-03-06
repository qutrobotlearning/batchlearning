from base  import *
import agent

AGENT     = agent.SelfSupervisedExplorationAgent
COMMENT   = "{}_{}".format(ENV, AGENT.__name__)
BATCH     = 64
SPIN_COST = 0.0
LR_M      = 2e-4
LR_P      = 1e-4
GAMMA     = 0.8
HORIZON   = 10
MODELS    = 2
ALPHA     = 1.0

DATA_OUT  = "{}_live".format(COMMENT)

