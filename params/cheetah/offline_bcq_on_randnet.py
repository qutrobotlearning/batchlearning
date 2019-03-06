from base import *
import agent

AGENT   = agent.BCQAgent
COMMENT = "{}_{}".format(ENV, AGENT.__name__)
LR_Q    = 1e-3
LR_P    = 1e-3
LR_VAE  = 1e-3
SIGMA   = 0.1
GAMMA   = 0.99
BATCH   = 100
D       = 1
TAU     = 0.005
NUM_Z   = 10
J_SCALE = 2
PSI     = 0.1
LAMBDA  = 0.75

UPDATES  = 10000
DATA_IN  = ".data/{}_RandomNetworksAgent_live.pytorch".format(ENV)

