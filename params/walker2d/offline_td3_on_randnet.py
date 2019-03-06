from base  import *
import agent

AGENT   = agent.TD3Agent
COMMENT = "{}_{}".format(ENV, AGENT.__name__)
LR_Q    = 1e-3
LR_P    = 1e-3
SIGMA   = 0.1
GAMMA   = 0.99
BATCH   = 100
D       = 1
TAU     = 0.005

UPDATES  = 300000
DATA_IN  = ".data/{}_RandomNetworksAgent_live.pytorch".format(ENV)

