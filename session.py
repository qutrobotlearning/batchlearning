
#! /usr/bin/env python2

from __future__ import print_function, division
import numpy as np, cv2, sys, os, math, random, time, torch, torch.nn as nn, collections, tensorboardX, gym, termcolor, datetime, colored_traceback.always
sys.dont_write_bytecode = True

from agent import Experience

#==============================================================================
# BASIC SESSION

class Session(object):
  def __init__(self, experience, writer, agent, env, task, run_tag, params, live):
    self.update     = 0
    self.episodes   = 0
    self.experience = experience
    self.writer     = writer
    self.agent      = agent
    self.env        = env
    self.task       = task
    self.run_tag    = run_tag
    self.params     = params
    self.live       = live

  #------------------------------------------------------------------------------

  def episode(self, train=True):
    metrics = {"episode/length": 0, "episode/reward_env": 0, "episode/reward_task": 0}
    metrics_reset_agent = self.agent.reset()
    metrics_reset_task  = self.task .reset()
    self._metrics("reset_agent", metrics_reset_agent, metrics)
    self._metrics("reset_task",  metrics_reset_task,  metrics)
    done = False
    obs  = self.env.reset()
    while not done:
      act = self.agent.act(self.task.obs(obs))
      nobs, rew, done, info = self.env.step(self._scale(act))
      step = self.task.online(Experience(obs, act, rew, nobs, done))
      if self.live: self.experience.append(step)
      metrics_train = self.agent.train(self.experience) if train else {}
      metrics_task  = self.task .train(self.experience) if train else {}
      self._metrics("actions",      {str(k):v for k,v in enumerate(act)}, metrics)
      self._metrics("observations", {str(k):v for k,v in enumerate(obs)}, metrics)
      self._metrics("train",        metrics_train, metrics)
      self._metrics("task",         metrics_task,  metrics)
      metrics["episode/reward_env" ] += rew
      metrics["episode/reward_task"] += step.rew
      metrics["episode/length"] += 1
      metrics["episode/buffer"] = len(self.experience)
      obs  = nobs
      done = step.done
    return metrics

  #------------------------------------------------------------------------------

  def run(self):
    while self.update < self.params.UPDATES:
      metrics        = self.episode()
      self.update   += metrics['episode/length']
      self.episodes += 1
      self.writer.add_scalar("episode/number", self.episodes, self.update)
      self.report(metrics)

    raw_experience = [self.task.decode(step) for step in self.experience]
    return raw_experience

  #------------------------------------------------------------------------------

  def report(self, metrics, step=None):
    if step is None: step = self.update
    for k,v in metrics.items():
      self.writer.add_scalar(str(k), np.mean(v), step)
      if sum(np.array(v).shape) > 1: # if we have more than just a scalar
        self.writer.add_scalar("{}_std".format(k), np.std(v), step)

  #------------------------------------------------------------------------------

  def _scale(self, act): return ((np.clip(act.astype(np.float32),-1,1)/2.0+0.5)*(self.env.action_space.high-self.env.action_space.low)+self.env.action_space.low).astype(np.float32)

  #------------------------------------------------------------------------------

  def _metrics(self, prefix, new, metrics):
    for k,v in new.items():
      if isinstance(v,torch.Tensor): v = v.detach().cpu().numpy()
      tag = "{}/{}".format(prefix, k)
      if tag not in metrics: metrics[tag] = []
      metrics[tag].append(v)

  #----------------------------------------------------------------------------

  def ckpt(self):
    if not os.path.exists(".ckpt"): os.mkdir(".ckpt")
    agent_state = self.agent.state_dict()
    task_state  = self.task .state_dict()
    torch.save(agent_state, ".ckpt/{}_{}_step{:07d}_agent.ckpt".format(self.params.COMMENT, self.params.DATA_IN.replace("/","_"), self.update))
    torch.save( task_state, ".ckpt/{}_{}_step{:07d}_task.ckpt" .format(self.params.COMMENT, self.params.DATA_IN.replace("/","_"), self.update))

  #----------------------------------------------------------------------------

  def load_ckpt(self):
    self.agent.load_state_dict(torch.load(self.params.CKPT_AGENT))
    self.task .load_state_dict(torch.load(self.params.CKPT_TASK))

#==============================================================================
# EVALUATION SESSION

class EvaluationSession(Session):
  def __init__(self, experience, writer, agent, env, task, run_tag, params, live):
    super(EvaluationSession, self).__init__(experience, writer, agent, env, task, run_tag, params, live)

    self.load_ckpt()
    self.episodes = 0

  #----------------------------------------------------------------------------

  def run(self):
    while self.episodes < self.params.EVAL_EPISODES:
      metrics = self.episode(train=False)
      self.episodes += 1
      self.report(metrics, step=self.episodes)

    return self.experience # because run assumes this

#==============================================================================
# OFFLINE SESSION

class OfflineSession(Session):
  def run(self):
    metrics = {}
    while self.update < self.params.UPDATES:
      metrics_train = self.agent.train(self.experience)
      metrics_task  = self.task .train(self.experience)
      self._metrics("train", metrics_train, metrics)
      self._metrics("task",  metrics_task,  metrics)
      if self.update %   1000 == 0: self.report(metrics); metrics = {}
      if self.update % 100000 == 0: self.ckpt()
      self.update  += 1
    self.report(metrics)
    self.ckpt()

    return self.experience # because run assumes this

