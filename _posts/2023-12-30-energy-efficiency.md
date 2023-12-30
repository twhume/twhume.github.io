---
layout: post
title:  "Energy Efficiency drives Predictive Coding in Neural Networks"
date:   2023-12-29
comments: true
---


I don’t remember how I came across it, but [this](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Predictive+coding+is+a+consequence+of+energy+efficiency+in+recurrent+neural+networks&btnG=) is one of the most exciting papers I’ve read recently. The authors train a neural network that tries to identify the next in a sequence of MNIST samples, presented in digit order. The interesting part is that when they include a proxy for energy usage in the loss function (i.e. train it to be more energy-efficient), the resulting network seems to exhibit the characteristics of predictive coding: some units seem to be responsible for predictions, others for encoding prediction error.

Why is this exciting?
* It proposes a plausible mechanism by which predictive coding would arise in practice.
* It shows an existence proof of this (well, two actually: one for MNIST and one for CIFAR images)
* It lines up artificial neural networks to theses around predictive coding from Andy Clark etc.

I grabbed the [source code](https://github.com/KietzmannLab/EmergentPredictiveCoding), tried to run it to replicate, and hit some issues (runtime errors etc), so have forked the repo to fix these, and also added support for the MPS backend (i.e. some acceleration on a Mac M1) which sped things up significantly - see my fork [here](https://github.com/twhume/EmergentPredictiveCoding/).

But lots of directions to go from here:
* I’d like to reimplement this in a framework like Jax, both to simplify it a little and to check I really understand it (and Jax)
* Does this approach work for more complex network architectures? For other tasks?

In the spirit of making it all run faster, I tried implementing early stopping (i.e. if you notice loss doesn’t keep falling, bail - on the basis you’ve found a local minima). Interestingly, it seemed that if I stopped too early (e.g. after just 5-10 epochs of loss not dropping) my results weren’t as good - i.e. the training process needs to really plug away at this fruitlessly for a while before it gets anywhere.
