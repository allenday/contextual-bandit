## Summary

This project contains a working example of a contextual multi-armed bandit.

I wrote this after becoming interested in the contextual bandit problem for
providing personalized recommendations, but not being able to find any working
code.  So I made this to be able to understand how the algorithm works :)

This repository indexes high on code and low on docs.  Briefly, it contains:

* A driver ipython notebook ``contextual_bandit_sim.ipynb``.  You should start
  here to understand the contents.

* A data generator.  We initialize hidden contextual variables that are used to
  create synthetic samples.  Let's call this latent contextual variable set ``L``.
  Let's call the synthetic data ``X``.

* Two strategies / reinforcement functions ``S`` by which the bandits can be
  evaluated (binomial: ``BinaryStrategy.py`` and continuous: ``PositiveStrategy.py``).

* A simulator ``Simulator.py`` that contains a set of variables ``M`` that are
  estimates of ``L``, that through observation of ``X`` and evaluation of those
  observations using stratigies ``S`` can be used to update estimates ``M`` to
  converge to the hidden ``L``.

If you find this useful, please share, [retweet](http://twitter.com/allenday/status/806414167062310912), and follow me [@allenday on twitter](http://twitter.com/allenday).

Have fun!
