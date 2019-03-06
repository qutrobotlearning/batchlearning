from base  import *
import agent

AGENT     = agent.GEPAgent
COMMENT   = "{}_{}".format(ENV, AGENT.__name__)

NUM_BOOT  = 50
NOISE_SIG = 0.01

DATA_OUT  = "{}_live".format(COMMENT)

