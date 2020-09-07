import tensorflow as tf
import numpy as np
import arviz as az

import pytest

import tensorflow_probability as tfp

tfd = tfp.distributions


def test_evaluate():
  from tfp_helper import evaluate
  assert (np.array([3., 4.]) == evaluate(tf.constant([3., 4.]))).all()


@pytest.fixture
def trace():
  return az.from_dict({ 'z' : np.random.normal(2., 10., (20,)) })


@pytest.fixture
def multi_var_trace():
  return az.from_dict(
      { 'z1' : np.random.normal(2., 10., (20,)),
        'z2' : np.random.normal(0., 1., (30,)),
        })


def test_plot_posterior_hist(trace):
  from tfp_helper import plot_posterior_hist
  plot_posterior_hist(trace, 'z', test_mode=True)


def test_az_to_dict(trace):
  from tfp_helper import az_to_dict
  trace_dict = az_to_dict(trace)
  assert trace_dict['z'].shape == (1, 20)


@pytest.fixture
def joint_log_prob():

  def fn(obs, p):
    var_p = tfd.Uniform(low=0., high=1.)
    var_obs = tfd.Bernoulli(probs=p)
    return (
        var_p.log_prob(p)
        + tf.reduce_sum(var_obs.log_prob(obs))
    )

  return fn


def test_infer(joint_log_prob):
  from tfp_helper import evaluate, infer
  obs = evaluate(tfd.Bernoulli(probs=0.75).sample(100))
  azdata = infer(joint_log_prob, obs,
              variables=['p'],
              initial_chain_state=[0.5],
              nsteps=5000, burn_in_ratio=0.8,
              bijectors={
                  'p' : tfp.bijectors.Sigmoid(),
              })
  pmean = azdata.posterior['p'].data.mean() 
  assert pmean > 0.65 and pmean < 0.85


"""
def test_az_to_numpy(multi_var_trace):
  from tfp_helper import az_to_numpy
  vars_ = az_to_numpy(multi_var_trace, flatten=False)
  print(vars_)
  assert len(vars_) == 2
  assert vars_[0].shape == (1, 20) and vars_[1].shape == (1, 30)
  
  # TODO BUG!! order is lost..
  vars_f = az_to_numpy(multi_var_trace, flatten=True)
  assert len(vars_f) == 2
  assert vars_f[0].shape == (20) and vars_f[1].shape == (30)
"""
