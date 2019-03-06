#! /usr/bin/env python2

from __future__ import print_function, division
import numpy as np, cv2, sys, os, math, random, time, torch, torch.nn as nn, collections, tensorboardX, gym
import colored_traceback.always
sys.dont_write_bytecode = True

#==============================================================================
# DATA STRUCTURES

Experience = collections.namedtuple("Experience", ("obs", "act", "rew", "nobs", "done"))

#==============================================================================
# BASE AGENT

class Agent(nn.Module):
  def __init__(self, task, live, params):
    super(Agent, self).__init__()
    self.task      = task
    self.live      = live
    self.params    = params
    self.obs_space = task.obs_space
    self.act_space = task.act_space
    self.discrete  = isinstance(self.act_space, gym.spaces.discrete.Discrete)
  #--------------------------------------------------------------------------
  def update_target(self, online, target, tau):
    for po,pt in zip(online.parameters(), target.parameters()): pt.data = tau*po.data.clone() + (1-tau)*pt.data.clone()
  #----------------------------------------------------------------------------
  def reset(self): return {}
  #----------------------------------------------------------------------------
  def act(self, obs):
    raise NotImplementedError("Agents must supply a self.act() method")
  #----------------------------------------------------------------------------
  def train(self, buf): return {}
  #----------------------------------------------------------------------------
  def batch(self, buf):
    batch = Experience(*map(lambda x: torch.FloatTensor(x).view(self.params.BATCH,-1).cuda(), zip(*random.sample(buf, self.params.BATCH))))
    batch = self.task.offline(batch)
    return batch

#==============================================================================
# RANDOM NETWORKS AGENT

class RandomNetworksAgent(Agent):
  def reset(self):
    self.pi = nn.Sequential(nn.Linear(self.obs_space.shape[0], self.act_space.shape[0]), nn.Tanh()).cuda()
    return {}
  #----------------------------------------------------------------------------
  def act(self, obs):
    act = self.pi(torch.FloatTensor(obs).view(1,-1).cuda()).detach().squeeze().cpu().numpy()
    return act

#==============================================================================
# GOAL-EXPLORATION-PROCESSES AGENT

class GEPAgent(Agent):
  def __init__(self, task, live, params):
    super(GEPAgent, self).__init__(task, live, params)
    self.num_resets = 0
    self.pi         = None
    self.library    = []
    self.trajectory = []

  #----------------------------------------------------------------------------

  def _descriptor(self, trajectory):
    return np.mean(trajectory, axis=0) # goal-agnostic trajectory descriptor. Original paper uses goal-specific features

  #----------------------------------------------------------------------------

  def reset(self):
    self.num_resets += 1
    if self.pi is not None:
      descriptor = self._descriptor(self.trajectory)
      network    = (self.pi[0].weight.data.cpu(), self.pi[0].bias.data.cpu())
      self.library.append((descriptor, network))

    if self.num_resets < self.params.NUM_BOOT:
      self.pi = nn.Sequential(nn.Linear(self.obs_space.shape[0], self.act_space.shape[0]), nn.Tanh()).cuda()
    else:
      goal = (np.random.random(self.obs_space.shape)*2-1)*np.pi
      descriptors = np.array([descriptor for descriptor, network in self.library])
      closest = np.argmin(np.linalg.norm(descriptors - goal, axis=1))
      descriptor, (weight, bias) = self.library[closest]
      self.pi = nn.Sequential(nn.Linear(self.obs_space.shape[0], self.act_space.shape[0]), nn.Tanh())
      self.pi[0].weight.data = weight + torch.randn(weight.shape)*self.params.NOISE_SIG
      self.pi[0].bias  .data = bias   + torch.randn(bias  .shape)*self.params.NOISE_SIG
      self.pi.cuda()

    return {}

  #----------------------------------------------------------------------------

  def act(self, obs):
    self.trajectory.append(obs.copy())
    act = self.pi(torch.FloatTensor(obs).view(1,-1).cuda()).detach().squeeze().cpu().numpy()
    return act

#==============================================================================
# SELF-SUPERVISED EXPLORATION AGENT

class SelfSupervisedExplorationAgent(Agent):
  def __init__(self, task, live, params):
    super(SelfSupervisedExplorationAgent, self).__init__(task, live, params)

    # networks
    self.pi_mu  = nn.Sequential(nn.Linear(self.obs_space.shape[0], 300), nn.ReLU(), nn.Linear(300, 400), nn.ReLU(), nn.Linear(400, self.act_space.shape[0]))
    self.pi_sig = nn.Sequential(nn.Linear(self.obs_space.shape[0], 300), nn.ReLU(), nn.Linear(300, 400), nn.ReLU(), nn.Linear(400, self.act_space.shape[0]), nn.Softplus())
    self.dh = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 300), nn.ReLU(), nn.Linear(300, 400), nn.ReLU(), nn.Linear(400, 1))
    self.ms = nn.ModuleList([nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 300), nn.ReLU(), nn.Linear(300, 400), nn.ReLU(), nn.Linear(400, self.obs_space.shape[0])) for m in range(self.params.MODELS)])

    # send it all to GPU
    self.cuda()

    # optimizers
    self.opt_pi = torch.optim.Adam(list(self.pi_mu.parameters()) + list(self.pi_sig.parameters()), lr=self.params.LR_P)
    self.opt_dh = torch.optim.Adam(self.dh.parameters(), lr=self.params.LR_M)
    self.opt_ms = [torch.optim.Adam(m.parameters(), lr=self.params.LR_M) for m in self.ms]

  #----------------------------------------------------------------------------

  def train_model(self, model, opt, label, buf):
    batch = self.batch(buf)

    nobs_pred = batch.obs + model(torch.cat([batch.obs, batch.act], dim=1))
    loss = torch.mean((nobs_pred - batch.nobs)**2)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return {"loss_m{}".format(label) : loss.detach()}

  #----------------------------------------------------------------------------

  def train_done(self, buf):
    batch = self.batch(buf)

    done_pred = self.dh(torch.cat([batch.obs, batch.act], dim=1))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(done_pred, batch.done)

    self.opt_dh.zero_grad()
    loss.backward()
    self.opt_dh.step()

    return {"loss_dh": loss.detach()}

  #----------------------------------------------------------------------------

  def sample_action(self, obs):
    mu   = self.pi_mu (obs)
    sig  = self.pi_sig(obs)
    dist = torch.distributions.Normal(loc=mu, scale=sig+1e-3)
    lgt  = dist.rsample()
    act  = torch.tanh(lgt)
    cor  = -torch.sum(torch.log(1 - act**2 + 1e-5), dim=1) # change of variables for PDF of tanh(gaussian)
    cor  = cor.unsqueeze(dim=1)
    lpi  = dist.log_prob(lgt) + cor
    lpi  = torch.sum(lpi, dim=1)
    return act, lpi

  #----------------------------------------------------------------------------

  def train_policy(self, buf):
    batch = self.batch(buf)

    prime_model_idx = np.random.randint(self.params.MODELS)

    loss_divergence = torch.zeros([self.params.MODELS, batch.obs.shape[0]]).cuda()
    loss_spin       = torch.zeros([]).float().cuda()
    loss_entropy    = torch.zeros([]).float().cuda()

    obs = batch.obs
    discount = torch.ones([self.params.MODELS, batch.obs.shape[0]]).float().cuda()
    for t in range(self.params.HORIZON):
      act, lpi = self.sample_action(obs)

      obs_act = torch.cat([obs, act], dim=1)

      obs_ms = [obs + m(obs_act) for m in self.ms]
      obs = obs_ms[prime_model_idx]

      # add up the divergence of the models
      divergences = torch.stack([torch.mean((obs_i - obs)**2, dim=1) for obs_i in obs_ms])

      # compute the discounted losses
      loss_divergence = loss_divergence - discount * divergences
      loss_spin       = loss_spin       + discount * self.params.SPIN_COST * torch.sum(act**2, dim=1)
      loss_entropy    = loss_entropy    + discount * self.params.ALPHA * lpi[None,:] # add a "model_id" dimension

      # estimate whether this action would end the episode, and discount accordingly
      done_h = torch.sigmoid(self.dh(obs_act)).squeeze(dim=1)[None,:] # add a "model_id" dimension
      discount = discount * self.params.GAMMA * (1 - done_h)

    # combine all timesteps of loss together, plus action regularizer
    loss_divergence = torch.mean(loss_divergence)
    loss_spin       = torch.mean(loss_spin)
    loss_entropy    = torch.mean(loss_entropy)
    loss = loss_divergence + loss_spin + loss_entropy
    loss = loss / self.params.HORIZON

    self.opt_pi.zero_grad()
    loss.backward()
    self.opt_pi.step()

    return {"loss_pi"           : loss           .detach(),
            "loss_pi_divergence": loss_divergence.detach(),
            "loss_pi_spin"      : loss_spin      .detach(),
            "loss_pi_entropy"   : loss_entropy   .detach()}

  #----------------------------------------------------------------------------

  def train(self, buf):
    if len(buf) < self.params.BATCH: return {}

    ms_metrics = [self.train_model(m, opt, i, buf).items() for i,(m,opt) in enumerate(zip(self.ms, self.opt_ms))]
    dh_metrics = self.train_done(buf)
    pi_metrics = self.train_policy(buf)

    return dict(reduce(lambda a,b: a+b, ms_metrics + [pi_metrics.items(), dh_metrics.items()], []))

  #----------------------------------------------------------------------------

  def act(self, obs):
    act = self.sample_action(torch.FloatTensor(obs).view(1,-1).cuda())[0].detach().squeeze().cpu().numpy()
    return act

#==============================================================================
# SOFT ACTOR CRITIC

class SoftACAgent(Agent):
  def __init__(self, task, live, params):
    super(SoftACAgent, self).__init__(task, live, params)

    # networks
    self.q1     = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300),  nn.ReLU(), nn.Linear(300, 1))
    self.q2     = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300),  nn.ReLU(), nn.Linear(300, 1))
    self.v      = nn.Sequential(nn.Linear(self.obs_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
    self.vt     = nn.Sequential(nn.Linear(self.obs_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
    self.pi_mu  = nn.Sequential(nn.Linear(self.obs_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, self.act_space.shape[0]))
    self.pi_sig = nn.Sequential(nn.Linear(self.obs_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, self.act_space.shape[0]), nn.ReLU())

    # send it all to GPU
    self.cuda()

    # separate optimizers
    self.opt_q1 = torch.optim.Adam(self.q1.parameters(), lr=self.params.LR)
    self.opt_q2 = torch.optim.Adam(self.q2.parameters(), lr=self.params.LR)
    self.opt_v  = torch.optim.Adam(self.v .parameters(), lr=self.params.LR)
    self.opt_pi = torch.optim.Adam(list(self.pi_mu.parameters()) + list(self.pi_sig.parameters()), lr=self.params.LR)

    # synchronize target value
    for po,pt in zip(self.v.parameters(), self.vt.parameters()):
      pt.data = po.data.clone()

  #--------------------------------------------------------------------------

  def sample_action(self, obs):
    mu  = self.pi_mu(obs)
    sig = self.pi_sig(obs)
    pi  = torch.distributions.Normal(mu, sig+self.params.SIGMA_EPS)
    lgt = pi.rsample()
    act = torch.tanh(lgt)
    cor = -torch.sum(torch.log(1 - act**2 + 1e-5), dim=1) # change of variables for PDF of tanh(gaussian)
    cor = cor.unsqueeze(dim=1)
    lpi = pi.log_prob(lgt) + cor
    lpi = lpi.unsqueeze(dim=1)
    lpi = self.params.ALPHA * lpi
    return act, lpi

  #-------------------------------------------------------------------------

  def train_val(self, buf):
    batch = self.batch(buf)

    # value learning
    opa, lpi = self.sample_action(batch.obs)
    q1  = self.q1(torch.cat([batch.obs, opa], dim=1))
    q2  = self.q2(torch.cat([batch.obs, opa], dim=1))
    qmn = torch.min(q1, q2)
    val = self.v(batch.obs)
    loss_v = torch.mean(0.5*(val - (qmn.detach() - lpi.detach()))**2)

    return {"loss_v": loss_v.detach()}, loss_v

  #--------------------------------------------------------------------------

  def train_q(self, buf, qnet, opt, n):
    batch = self.batch(buf)

    # q learning
    vtn = self.vt(batch.nobs)
    qt  = self.params.RSCALE * batch.rew + self.params.GAMMA*vtn * (1-batch.done)
    q   = qnet(torch.cat([batch.obs, batch.act], dim=1))
    loss_q = torch.mean(0.5*(q - qt.detach())**2)

    return {"loss_q{}".format(n): loss_q.detach()}, loss_q

  #--------------------------------------------------------------------------

  def train_pi(self, buf):
    batch = self.batch(buf)

    # policy learning
    opa, lpi = self.sample_action(batch.obs)
    q1  = self.q1(torch.cat([batch.obs, opa], dim=1))
    q2  = self.q2(torch.cat([batch.obs, opa], dim=1))
    qmn = torch.min(q1, q2)
    loss_pi = torch.mean(lpi - qmn)

    return {"loss_pi": loss_pi.detach(),
            "lpi"    : lpi    .detach(),
            "q1"     :  q1    .detach(),
            "q2"     :  q2    .detach(),
            "qmn"    : qmn    .detach(),
            "act"    : opa    .detach(),
           }, loss_pi

  #--------------------------------------------------------------------------

  def train(self, buf):
    if len(buf) < self.params.BATCH: return {}
    metrics = {}

    for gradient_step in range(self.params.STEPS):
      metrics_v,  loss_v  = self.train_val(buf)
      metrics_q1, loss_q1 = self.train_q(buf, self.q1, self.opt_q1, 1)
      metrics_q2, loss_q2 = self.train_q(buf, self.q2, self.opt_q2, 2)
      metrics_pi, loss_pi = self.train_pi(buf)
      metrics = dict(metrics_v.items() + metrics_q1.items() + metrics_q2.items() + metrics_pi.items())

      for loss, opt in [(loss_v, self.opt_v), (loss_q1, self.opt_q1), (loss_q2, self.opt_q2), (loss_pi, self.opt_pi)]:
        opt.zero_grad()
        loss.backward()
        opt.step()

      for po,pt in zip(self.v.parameters(), self.vt.parameters()):
        pt.data = (1-self.params.TAU)*pt.data + self.params.TAU*po.data

    return metrics

  #--------------------------------------------------------------------------

  def act(self, obs):
    obs = torch.from_numpy(obs).float().view(1,-1).cuda()
    act, lpi = self.sample_action(obs)
    act = act.detach().squeeze().cpu().numpy()
    return act

#==============================================================================
# TD3 AGENT

class TD3Agent(Agent):
  def __init__(self, task, live, params):
    super(TD3Agent, self).__init__(task, live, params)

    # networks
    self.q1  = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
    self.q2  = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
    self.pi  = nn.Sequential(nn.Linear(self.obs_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, self.act_space.shape[0]), nn.Tanh())

    # targets
    self.q1t = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
    self.q2t = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
    self.pit = nn.Sequential(nn.Linear(self.obs_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, self.act_space.shape[0]), nn.Tanh())

    # send it all to the GPU
    self.cuda()

    # hard sync the targets
    self.update_target(self.q1, self.q1t, 1.0)
    self.update_target(self.q2, self.q2t, 1.0)
    self.update_target(self.pi, self.pit, 1.0)

    # optimizers
    self.oq1 = torch.optim.Adam(self.q1.parameters(), lr=self.params.LR_Q)
    self.oq2 = torch.optim.Adam(self.q2.parameters(), lr=self.params.LR_Q)
    self.opi = torch.optim.Adam(self.pi.parameters(), lr=self.params.LR_P)

    # keep track of how many steps we've taken
    self.update = 0

  #----------------------------------------------------------------------------

  def _noise(self): return torch.clamp(torch.randn(self.act_space.shape[0]).cuda() * self.params.SIGMA, -1, 1)

  #----------------------------------------------------------------------------

  def train_q(self, q, opt, label, buf):
    batch = self.batch(buf)

    qt  = q(torch.cat([batch.obs, batch.act], dim=1))
    an  = torch.clamp(self.pi(batch.nobs) + self._noise(), -1, 1)
    qn1 = self.q1t(torch.cat([batch.nobs, an], dim=1))
    qn2 = self.q2t(torch.cat([batch.nobs, an], dim=1))
    qn  = torch.min(qn1, qn2)
    y   = batch.rew + self.params.GAMMA * (1-batch.done) * qn.detach()

    loss = torch.mean((qt - y)**2)
    opt.zero_grad()
    loss.backward()
    opt.step()

    return {"q{}_qt"  .format(label): qt  .detach(),
            "q{}_an"  .format(label): an  .detach(),
            "q{}_qn"  .format(label): qn  .detach(),
            "q{}_y"   .format(label): y   .detach(),
            "q{}_loss".format(label): loss.detach()}

  #----------------------------------------------------------------------------

  def train_pi(self, buf):
    if self.update % self.params.D != 0: return {}
    batch = self.batch(buf)

    act  = self.pi(batch.obs)
    qt   = self.q1(torch.cat([batch.obs, act], dim=1))

    loss = -torch.mean(qt)
    self.opi.zero_grad()
    loss.backward()
    self.opi.step()

    self.update_target(self.q1, self.q1t, self.params.TAU)
    self.update_target(self.q2, self.q2t, self.params.TAU)
    self.update_target(self.pi, self.pit, self.params.TAU)

    return {"pi_act" : act .detach(),
            "pi_qt"  : qt  .detach(),
            "pi_loss": loss.detach()}

  #----------------------------------------------------------------------------

  def train(self, buf):
    if len(buf) < self.params.BATCH: return {}

    metrics_q1 = self.train_q (self.q1, self.oq1, "1", buf)
    metrics_q2 = self.train_q (self.q2, self.oq2, "2", buf)
    metrics_pi = self.train_pi(buf)
    self.update += 1

    return dict(metrics_q1.items() + metrics_q2.items() + metrics_pi.items())

  #----------------------------------------------------------------------------

  def act(self, obs):
    obs = torch.FloatTensor(obs.astype(np.float32)).view(1,-1).cuda()
    act = (self.pi(obs) + self._noise()).detach().squeeze().cpu().numpy()
    return act

#==============================================================================
# BCQ AGENT

class BCQAgent(Agent):
  def __init__(self, task, live, params):
    super(BCQAgent, self).__init__(task, live, params)

    self.J = self.params.J_SCALE * self.act_space.shape[0]

    # networks
    self.q1  = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
    self.q2  = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))

    # targets
    self.q1t = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
    self.q2t = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))

    # VAE
    self.enc  = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 750), nn.ReLU(), nn.Linear(750, 750), nn.ReLU())
    self.mu   = nn.Linear(750, self.J)
    self.sig  = nn.Linear(750, self.J)
    self.dec = nn.Sequential(nn.Linear(self.J+self.obs_space.shape[0], 750), nn.ReLU(), nn.Linear(750, 750), nn.ReLU(), nn.Linear(750, self.act_space.shape[0]))

    # perturbation network
    self.pert = nn.Sequential(nn.Linear(self.obs_space.shape[0]+self.act_space.shape[0], 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, self.act_space.shape[0]), nn.Tanh())

    # hard sync the targets
    self.update_target(self.q1, self.q1t, 1.0)
    self.update_target(self.q2, self.q2t, 1.0)

    # GPU
    self.cuda()

    # optimizers
    self.oq1 = torch.optim.Adam(self.q1.parameters(), lr=self.params.LR_Q)
    self.oq2 = torch.optim.Adam(self.q2.parameters(), lr=self.params.LR_Q)
    self.opt_vae  = torch.optim.Adam(list(self.mu.parameters()) + list(self.sig.parameters()) + list(self.enc.parameters()) + list(self.dec.parameters()), lr=self.params.LR_VAE)
    self.opt_pert = torch.optim.Adam(self.pert.parameters(), lr=self.params.LR_P)

    self.update = 0

  #------------------------------------------------------------------------------

  def eval_act(self, obs):
    # standard normal
    dist = torch.distributions.Normal(torch.zeros((obs.shape[0],self.J)), torch.ones((obs.shape[0],self.J)))
    # sample 10 times, clamp to ~half sigma
    samples = dist.rsample((self.params.NUM_Z,))#.view(self.params.NUM_Z*obs.shape[0], self.J)
    z = torch.clamp(samples, -0.5, 0.5).cuda()
    # get action from policy and perturb net
    expanded_obs = obs.unsqueeze(0).expand((self.params.NUM_Z, obs.shape[0], obs.shape[1]))
    act = self.dec(torch.cat([expanded_obs, z], dim=2))
    act = act + self.params.PSI*self.pert(torch.cat([expanded_obs, act], dim=2))
    return act

  #------------------------------------------------------------------------------

  def train_q(self, q, opt, label, buf):
    batch = self.batch(buf)
    expanded_nobs = batch.nobs.unsqueeze(0).expand((self.params.NUM_Z, batch.nobs.shape[0], batch.nobs.shape[1]))

    qt  = q(torch.cat([batch.obs, batch.act], dim=1))
    an  = torch.clamp(self.eval_act(batch.nobs), -1, 1)
    qn1 = self.q1t(torch.cat([expanded_nobs, an], dim=2))
    qn2 = self.q2t(torch.cat([expanded_nobs, an], dim=2))

    qmnb = torch.min(qn1, qn2)
    qmxb = torch.max(qn1, qn2)
    qn_min, _ = torch.max(qmnb, dim=0)
    qn_max, _ = torch.max(qmxb, dim=0)

    qn = self.params.LAMBDA * qn_min + (1-self.params.LAMBDA)*qn_max
    y   = batch.rew + self.params.GAMMA * (1-batch.done) * qn.detach()

    loss = torch.mean((qt - y)**2)
    opt.zero_grad()
    loss.backward()
    opt.step()

    return {"q{}_qt"  .format(label): qt  .detach(),
            "q{}_an"  .format(label): an  .detach(),
            "q{}_qn"  .format(label): qn  .detach(),
            "q{}_y"   .format(label): y   .detach(),
            "q{}_loss".format(label): loss.detach()}

  #------------------------------------------------------------------------------

  def train_pert(self, buf):
    if self.update % self.params.D != 0: return {}
    batch = self.batch(buf)

    act = batch.act + self.params.PSI*self.pert(torch.cat([batch.obs, batch.act], dim=1))
    qt   = self.q1(torch.cat([batch.obs, act], dim=1))
    loss = -torch.mean(qt)
    self.opt_pert.zero_grad()
    loss.backward()
    self.opt_pert.step()

    self.update_target(self.q1, self.q1t, self.params.TAU)
    self.update_target(self.q2, self.q2t, self.params.TAU)

    return {"pi_act" : act .detach(),
            "pi_qt"  : qt  .detach(),
            "pi_loss": loss.detach()}

  #------------------------------------------------------------------------------

  def train_vae(self, buf):
    batch = self.batch(buf)

    emb = self.enc(torch.cat([batch.obs, batch.act], dim=1))
    mu  = self.mu(emb)
    sig = self.sig(emb)

    dist  = torch.distributions.Normal(mu, sig)
    z     = dist.rsample()
    act_h = self.dec(torch.cat([batch.obs, z], dim=1))

    prior = torch.distributions.Normal(torch.zeros(mu.shape).cuda(), torch.ones(sig.shape).cuda())
    loss = torch.mean((batch.act - act_h)**2) + torch.mean(torch.distributions.kl.kl_divergence(dist, prior))/(self.J*2)

    self.opt_vae.zero_grad()
    loss.backward()
    self.opt_vae.step()

    return {"vae_loss": loss.detach()}

  #----------------------------------------------------------------------------

  def train(self, buf):
    if len(buf) < self.params.BATCH: return {}

    metrics_q1 = self.train_q(self.q1, self.oq1, "1", buf)
    metrics_q2 = self.train_q(self.q2, self.oq2, "2", buf)
    metrics_pert = self.train_pert(buf)
    metrics_vae  = self.train_vae(buf)
    self.update += 1

    return dict(metrics_q1.items() + metrics_q2.items() + metrics_pert.items() + metrics_vae.items())

  #----------------------------------------------------------------------------

  def act(self, obs):
    obs = torch.FloatTensor(obs.astype(np.float32)).view(1,-1).cuda()
    act = self.eval_act(obs)
    a  = torch.clamp(self.eval_act(obs), -1, 1)
    expanded_obs = obs.unsqueeze(0).expand((self.params.NUM_Z, obs.shape[0], obs.shape[1]))
    qn1 = self.q1t(torch.cat([expanded_obs, a], dim=2))
    qn1, qn_i = torch.max(qn1, dim=0)
    act = a[qn_i.squeeze().item()]
    act = act + self.params.PSI*self.pert(torch.cat([obs, act], dim=1))
    act = act.detach().squeeze().cpu().numpy()
    return act

