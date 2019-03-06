
#! /usr/bin/env python2

from __future__ import print_function, division
import numpy as np, cv2, sys, os, math, random, time, torch, torch.nn as nn, collections, tensorboardX, gym, termcolor, datetime, colored_traceback.always
sys.dont_write_bytecode = True

from agent import Experience

#==============================================================================
# BASE TASK

class Task(nn.Module):
  def __init__(self, obs_space, act_space, params):
    super(Task, self).__init__()
    self.obs_space = obs_space
    self.act_space = act_space
    self.params    = params
  def reset(self): return {}
  def obs(self, obs): return obs
  def online(self, step): return step
  def offline(self, batch): return batch
  def decode(self, step): return step
  def batch(self, buf): return Experience(*map(lambda x: torch.FloatTensor(x).view(self.params.BATCH,-1).cuda(), zip(*random.sample(buf, self.params.BATCH))))
  def train(self, buf): return {}

#==============================================================================
# RND TASK

class RNDTask(Task):
  def __init__(self, obs_space, act_space, params):
    super(RNDTask, self).__init__(obs_space, act_space, params)
    # networks
    self.targn = nn.Sequential(nn.Linear(obs_space.shape[0], 64), nn.ReLU(), nn.Linear(64, 64))
    self.predn = nn.Sequential(nn.Linear(obs_space.shape[0], 64), nn.ReLU(), nn.Linear(64, 64))
    # gpu
    self.cuda()
     # optimizers
    self.opt_p = torch.optim.Adam(self.predn.parameters(), lr=self.params.LR_RND)

    self.updates = 0

  #----------------------------------------------------------------------------

  def offline(self, batch):
    rew = torch.sum((self.targn(batch.obs) - self.predn(batch.obs))**2, dim=1)
    batch = Experience(batch.obs, batch.act, rew.detach(), batch.nobs, batch.done)
    return batch

  #----------------------------------------------------------------------------

  def train(self, buf):
    if len(buf) < self.params.BATCH: return {}
    batch = self.batch(buf)
    loss = torch.mean((self.targn(batch.obs) - self.predn(batch.obs))**2)
    self.opt_p.zero_grad()
    loss.backward()
    self.opt_p.step()
    self.updates += 1
    return {"loss_pred": loss.detach()}

#==============================================================================
# DIAYN TASK

class DIAYNTask(Task):
  def __init__(self, obs_space, act_space, params):
    super(DIAYNTask, self).__init__(obs_space, act_space, params)

    # network
    self.discr = nn.Sequential(nn.Linear(obs_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, self.params.L_SIZE))
    # send to GPU
    self.cuda()
    # optimizer
    self.opt_d  = torch.optim.Adam(self.discr.parameters(), lr=self.params.LR_D)

    # augment the observation space with the latent code
    self.obs_space = gym.spaces.Box(shape=(obs_space.shape[0] + self.params.L_SIZE,), low=self.obs_space.low[0], high=self.obs_space.high[0])

  #----------------------------------------------------------------------------

  def reset(self):
    self.z = np.random.randint(self.params.L_SIZE)
    return {"skill": self.z}

  #----------------------------------------------------------------------------

  def obs(self, obs):
    latent = np.zeros((self.params.L_SIZE,), np.float32); latent[self.z] = 1
    obs    = np.concatenate([obs, latent], axis=0)
    return obs

  #----------------------------------------------------------------------------

  def decode(self, step):
    pruned_obs  = step.obs [:-self.params.L_SIZE]
    pruned_nobs = step.nobs[:-self.params.L_SIZE]
    return Experience(pruned_obs, step.act, step.rew, pruned_nobs, step.done)

  #----------------------------------------------------------------------------

  def _from_one_hot(self, latent):
    _, idx = torch.max(latent, dim=1)
    return idx

  #----------------------------------------------------------------------------

  def online(self, step):
    return Experience(self.obs(step.obs), step.act, step.rew, self.obs(step.nobs), step.done)

  #----------------------------------------------------------------------------

  def offline(self, batch):
    pruned_nobs = batch.nobs[:,:-self.params.L_SIZE]
    logits = self.discr(pruned_nobs)
    target = self._from_one_hot(batch.nobs[:,-self.params.L_SIZE:])
    rew = -torch.nn.functional.cross_entropy(logits, target, reduction="none") - np.log(1.0/self.params.L_SIZE)
    batch = Experience(batch.obs, batch.act, rew.detach().unsqueeze(1), batch.nobs, batch.done)
    return batch

  #----------------------------------------------------------------------------

  def train(self, buf):
    if len(buf) < self.params.BATCH: return {}
    batch  = self.batch(buf)
    pruned_nobs = batch.nobs[:,:-self.params.L_SIZE]
    logits = self.discr(pruned_nobs)
    target = self._from_one_hot(batch.nobs[:,-self.params.L_SIZE:])
    loss   = torch.nn.functional.cross_entropy(logits, target)
    self.opt_d.zero_grad()
    loss.backward()
    self.opt_d.step()
    return {"loss_diayn_disc": loss.detach()}

#==============================================================================
# PANDA REACHING TASK

class PandaReachingTask(Task):
  def __init__(self, obs_space, act_space, params):
    super(PandaReachingTask, self).__init__(obs_space, act_space, params)

    self.obs_space = gym.spaces.Box(shape=(14,), low=-np.pi, high=np.pi)

  #------------------------------------------------------------------------------

  def obs(self, obs):
    return obs[:-3]

  #------------------------------------------------------------------------------

  def reset(self):
    self.time = 0
    return {}

  #------------------------------------------------------------------------------

  def online(self, step):
    self.time += 1
    goal = np.array(self.params.TASK_GOAL).astype(np.float32)
    dist = np.linalg.norm(step.nobs[-3:] - goal)
    rew  = -dist
    done = dist < self.params.TASK_THRESHOLD or self.time >= self.params.TASK_TIME_LIMIT or step.done
    step = Experience(step.obs, step.act, rew, step.nobs, done)
    return step

  #------------------------------------------------------------------------------

  def _batch_rew(self, nobs):
    goal = torch.from_numpy(np.array(self.params.TASK_GOAL).astype(np.float32)).cuda()
    dist = torch.norm(nobs[:,-3:] - goal, dim=1)
    rew  = -dist
    return rew

  #------------------------------------------------------------------------------

  def _batch_obs(self, obs):
    return obs[:,:-3]

  #------------------------------------------------------------------------------

  def _batch_done(self, nobs):
    goal = torch.from_numpy(np.array(self.params.TASK_GOAL).astype(np.float32)).cuda()
    dist = torch.norm(nobs[:,-3:] - goal, dim=1)
    done = (dist < self.params.TASK_THRESHOLD).float()
    return done

  #------------------------------------------------------------------------------

  def offline(self, batch):
    obs   = self._batch_obs (batch.obs)
    rew   = self._batch_rew (batch.nobs)
    nobs  = self._batch_obs (batch.nobs)
    done  = self._batch_done(batch.nobs)
    batch = Experience(obs, batch.act, rew, nobs, done)
    return batch

