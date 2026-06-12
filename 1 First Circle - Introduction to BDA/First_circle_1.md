---
author: Jiří Fejlek
bibliography: First_circle_1.bib
code_folding: hide
date: 2025-05-25
output:
  md_document:
    toc: true
    variant: GFM
title: "The First Circle: Introduction to Bayesian Statistics"
---

- [Bayesian Data Analysis](#bayesian-data-analysis)
- [Single-parameter models](#single-parameter-models)
  - [Binomial data](#binomial-data)

## Bayesian Data Analysis

why Bayesian? measurement errors, multilevel models, missing data,
regularization, latent variable, generative

Three main steps

1.  Setting up a *full probability model*, a joint probability
    distribution for all observable and unobservable quantities
2.  Computing *posterior distribution* of the unobserved quantities of
    interest given the observed data
3.  Evaluating the fit of the model and performing inference based on
    the resulting posterior distribution

$\theta$ unobservable quantities (typically population parameters such
as survival probabilities), $y$ observed data, $\tilde y$ potentially
observable quantities/outcomes (such as future observations)

Bayes rule:

definition of conditional probability
$$p(y, \theta) = p(y \mid \theta)p(\theta)$$ implies
$$p(\theta \mid y) = \frac{p(y, \theta)}{p(y)} = \frac{p(y \mid \theta)p(\theta)}{\int p(y \mid \theta)p(\theta) \mathrm{d}\theta}$$
$p(y \mid \theta)$ is *likelihood* function -\> posterior probability
(and thus Bayes inference) for $\theta$ depends on the observed data $y$
only through the likelihood function -\> *likelihood principle*

bayesian update: prior -\> posterior -\> prior for a next update -\>
Bayes naturally process se no approx -\> no minimum samle size, shape of
posterior embodies samples size (no DOF), no point estimates -\>
estimate is the posterior distribution, no one true interval

*prior predictive predictions* (before data $y$ are gathered)

$$ p(y) = \int p(y, \theta) \mathrm{d}\theta = \int p(y \mid \theta) p(\theta) \mathrm{d}\theta$$

*posterior predictive distribution*: we gather $y$ and want to predict
new observable $\tilde y$:

 $p(\tilde y \mid y) = \int p(\tilde y, \theta \mid y) \mathrm{d}\theta = \int p(\tilde y \mid y, \theta) p(\theta \mid y)p(\theta) \mathrm{d}\theta = \int p(\tilde y \mid \theta) p(\theta \mid y)p(\theta) \mathrm{d}\theta $
we used in the last derivation that $\tilde y$ is conditionally
independent of $y$

## Single-parameter models

### Binomial data

sequence of “Bernoulli trials”, binomial sapling model is
$$ p(y \mid \theta) = \binom{n}{y} \theta^y (1-\theta)^{n-y}$$

conditioning on observed data: calculating and interpreting the
appropriate posterior distribution—the conditional probability
distribution of the unobserved quantities of ul- timate interest, given
the observed data.

Prior prediction!
