---
layout: post
title:  "Weakly trained ensembles of neural networks"
date:   2021-04-24
comments: true
---


# What if neural networks, but not very good and lots of them?

I recently read [A Thousand Brains](https://www.amazon.com/Thousand-Brains-New-Theory-Intelligence/dp/1541675819), by Jeff Hawkins. I’ve been a fan of his since [reading the deeply inspirational founding stories](https://www.amazon.com/Information-Appliances-Beyond-Interaction-Technologies/dp/1558606009) of the Palm Pilot (original team: 7 people! 7!). And as an AI weenie when I discovered what he **really** wanted to do all along was understand intelligence, I was doubly impressed - and loved his first book, [On Intelligence](https://www.amazon.com/Intelligence-Understanding-Creation-Intelligent-Machines/dp/0805078533).

Paraphrasing one of the theses of A Thousand Brains: the neocortex is more-or-less uniform and composed of \~150k cortical columns of equivalent structure, wired into different parts of the sensorimotor system. Hawkins suggests that they all build models of the world based on their inputs - so models are duplicated throughout the neocortex - and effectively “vote” to agree on the state of the world. So, for instance, if I’m looking at a coffee cup whilst holding it in my hand, columns receiving visual input and touch input each separately vote “coffee cup”. 

The notion of many agents coming to consensus is also prominent in Minsky’s [Society of Mind](https://www.amazon.com/Society-Mind-Marvin-Minsky/dp/0671657135), and in the [Copycat architecture](https://en.wikipedia.org/wiki/Copycat_(software)) which suffuses much of Hofstadter’s work. I spent a bit of time last year [noodling](http://tomhume.org/numbo/) with the latter.

We know the brain doesn’t do backprop (in the same way as neural networks do - Hinton [suggests](https://syncedreview.com/2020/04/23/new-hinton-nature-paper-revisits-backpropagation-offers-insights-for-understanding-learning-in-the-cortex/) it might do something similar, and some folks from Sussex and Edinburgh have recently proposed [how this might work](https://arxiv.org/abs/2006.04182)). We do know the brain is massively parallel, and that it can learn quickly from small data sets.

This had me wondering how classical neural networks might behave, if deployed in large numbers (typically called “ensembles”) that vote after being trained weakly - as opposed to being trained all the way to accurate classification in a single network. Could enormous parallelism compensate in some way for either training time or dataset size?

# Past work with ensembles

One thing my Master’s (pre-Google) gave me an appreciation for was digging into academic literature. I had a quick look around to see what’s been done previously - there’s no real technical innovation for this kind of exploration, so I expected to find it thoroughly examined. Here’s what I found.

Large ensembles haven’t been deeply explored:

* [Lincoln and Skrzypek](https://scholar.google.com/scholar?cluster=8267063715713827199&hl=en&as_sdt=0,5) trained an ensemble of 5 networks and observed better performance over a single network.
* [Donini, Loreggia, Pini and Rossi](http://ceur-ws.org/Vol-2272/short6.pdf) did experiments with 10.
* [Breiman](https://scholar.google.com/scholar?cluster=18412826781870444603&hl=en&as_sdt=0,5) got to 25 (*"more than 25 bootstrap replicates is love’s labor lost”*), as did [Opitz and Maclin](https://scholar.google.com/scholar?cluster=9297768410353397265&hl=en&as_sdt=0,5) 

Voting turns up in many places:

*  [Drucker et al](https://direct.mit.edu/neco/article-pdf/6/6/1289/812901/neco.1994.6.6.1289.pdf) point to many studies of neural networks in committee, initialized with different weights, and suggest an expectation I share: that multiple networks might converge differently and thus have performance improved in combination. They had explored this in previous work (Drucker et al 1993a).
* [Donini, Loreggia, Pini and Rossi](http://ceur-ws.org/Vol-2272/short6.pdf) reach the same conclusion and articulate it thus: *“different neural networks can generalize the learning functions in different ways as to learn different areas of the solutions space”*. They also explore other voting schemes.
* [Clemen](https://faculty.fuqua.duke.edu/~clemen/bio/Published%20Papers/13.CombiningReview-Clemen-IJOF-89.pdf)  notes that *“simple combination methods often work reasonably well relative to more complex combinations”*, generally in forecasting.
* [Kittler, Duin and Matas](https://dspace.cvut.cz/bitstream/handle/10467/9443/1998-On-combining-classifiers.pdf?sequence=1) similarly note that *“the sum rules out performs other classifier combinations schemes"*.


Breiman’s [Bagging](https://scholar.google.com/scholar?cluster=18412826781870444603&hl=en&as_sdt=0,5) (short for “bootstrap aggregating”) seems very relevant. It involves sampling from the same distribution and training a different network on each sample, then combining their results via voting etc. Technically if you sampled fully from that distribution and thus trained all your ensemble of the same dataset, this would be bagging, but it seems a bit over-literal and incompatible in spirit.

In his paper Breiman does not explore the impact of parallelization on small dataset sizes, but does note that *“bagging is almost a dream procedure for parallel computing”*, given the lack of communication between predictors.

Finally, two other titbits stood out:

1. [Perrone and Cooper](https://apps.dtic.mil/sti/pdfs/ADA260045.pdf) observe that training on hold-out data is possible with an ensemble process without risking overfitting, as we *“let the smoothing property of the ensemble process remove any overfitting or we can train each network with a different split of training and hold-out data”*. That seems interesting from the POV of maximizing the value of a small dataset.
2. [Krogh and Vedelsby](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.52.9672&rep=rep1&type=pdf) have an interesting means to formalize a measure of ambiguity in an ensemble. [Opitz and Maclin](https://www.jair.org/index.php/jair/article/download/10239/24370/) reference them as verifying that classifieds which disagree strongly perform better. I wondered if this means an ensemble mixing networks designed to optimize for individual class recognition might perform better than multiclass classifiers?

# Questions to ask

This left me wanting to answer a few questions

1. What’s the trade-off between dataset size and ensemble size? i.e. would parallelization help a system compensate for having very few training examples, and/or very little training effort?
2. Is an ensemble designed to do multiclass classification best served by being formed of homogenous networks, or specialist individual classifiers?
3. How might we best optimize the performance of an ensemble?


I chose a classic toy problem, MNIST digit classification, and worked using the [Apache MXNet](https://mxnet.apache.org/versions/1.8.0/) framework. I chose the latter for a very poor reason: I started out wanting to use Clojure because I enjoy writing it, and MXNet seemed like the best option…. but I struggled to get it working and switched to Python.

The MNIST dataset has 60,000 images and MXNet is bundled with [a simple neural network](https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/packages/gluon/image/mnist.html) for it: three fully connected layers with 128, 64 and 10 neurons each, which get 98% accuracy after training for 10 epochs (i.e. seeing 600,000 images total).

## What’s the trade-off between dataset and ensemble size?

I started like this:

1. Taking a small slice of the MNIST data set (100-1000 images out of the 60,000), to approximate the small number of examples a human might see.
2. Training on this small dataset for a very small number of epochs (1 to 10), to reflect the fact that humans don’t learn by repeated presentation.
3. Repeating the training on 10,000 copies of a simple neural network.
4. For increasingly sized subsets of these 10,000 networks, having them vote on what they felt the most likely outcome was, and taking the most voted result as the classification. I tested on subsets to try and understand where the diminishing returns for parallelization might be: 100 networks? 1000?

I ran everything serially, so the time to train and time to return a classification were extremely long: in a truly parallel system they’d likely be 1/10,000th.

I ran two sets of tests:

1. With dataset sizes of 200, 500, 1000 and 10000 examples from the MNIST set, all trained for a single epoch. I also ran a test with a completely untrained network that had seen no data at all, to act as a baseline.
2. For a dataset of 200 examples, I tried training for 1, 10, and 100 epochs.

It’s worth reiterating: these are **very** small training data sets (even 10,000 is 1/6th of the MNIST data set).

I expected to see increased performance from larger data sets, and from more training done on the same data set, but I had no intuition over how far I could go (I assumed a ceiling of 0.98, given this is where a well-trained version of the MXNet model got to).

I hoped to see increased performance from larger ensembles I had no intuition about how far this could go.

I expected the untrained model to remain at 0.1 accuracy no matter how large the ensemble, on the basis that it could not have learned anything about the data set, so its guesses would be effectively random.

### Results

For dataset sizes trained for a single epoch [here](https://docs.google.com/spreadsheets/d/1-fzEgBgxJqEy1jdDRAnopiLNtJvPpQACtkgAzQ0l2nU/edit#gid=1179630915):

![Graph showing effect of training data size on classification accuracy](https://docs.google.com/spreadsheets/d/e/2PACX-1vSXwJIn0X6AcgyRXjztNHnkImcbX-EgGmLbMinxy9o69WP0D03A_3BMF3_WjJ9i42f5Zp8nz9Ypl4Ld/pubchart?oid=318615433&format=image)

Interpreting:

* A larger dataset leads to improved accuracy, and faster arrival at peak accuracy: for d=10000 1 network scored 0.705, an ensemble of 50 scored 0.733 and by 300, the ensemble converged on 0.741 where it remained.
* For smaller datasets, parallelization continues to deliver benefits  for some time: d=200 didn’t converge near 0.47 (its final accuracy being 0.477) until an ensemble of \~6500 networks.
* An untrained network still saw slight performance improvements (0.0088 with 1 network, to the 0.14 range by 6000.

Looking at the impact of training time (in number of epochs) ([data](https://docs.google.com/spreadsheets/d/1-fzEgBgxJqEy1jdDRAnopiLNtJvPpQACtkgAzQ0l2nU/edit#gid=1235499981)):

![Graph showing effect of training time on classification accuracy](https://docs.google.com/spreadsheets/d/e/2PACX-1vSXwJIn0X6AcgyRXjztNHnkImcbX-EgGmLbMinxy9o69WP0D03A_3BMF3_WjJ9i42f5Zp8nz9Ypl4Ld/pubchart?oid=1860813187&format=image)

Interpreting:

* **More training means less value from an ensemble**: 100 rose from 0.646 accuracy with 1 network to 0.667 by 50 networks, and stayed there.
* **Less training means more value from an ensemble**: 1 epoch rose from 0.091 accuracy to 0.55 by the time the ensemble reached 4500 networks.

Conclusions here:

1. **Parallelization can indeed compensate for either a small dataset or less time training, but not fully**: an ensemble trained on 10,000 examples scored 0.744 vs 0.477 for one trained on 200; one trained for 100 epochs scored 0.668 vs 0.477 for one trained for 1 epoch.
2. I don’t understand how an untrained network gets better. Is it reflecting some bias in the training/validation data, perhaps? i.e. learning that there are slightly more examples of the digit 1 than 7 etc?

## Should individual classifiers be homogenous or heterogeneous?

Instead of training all networks in the ensemble on all classes, I moved to a model where each network was trained on a single class.

To distinguish a network’s target class from others, I experimented with different ratios of true:false training data in the training set (2:1, 1:1, 1:2, and 1:3)

I took the best performing ratio and tried it with ensembles of various sizes, trained for a single epoch on data sets of different sizes. I then compared these to the homogenous networks I'd been using previously.

And finally, I tried different data set sizes with well-trained ensembles, each network being trained for 100 epochs.

### Results

Here’s a comparison of those data ratios ([data](https://docs.google.com/spreadsheets/d/1-fzEgBgxJqEy1jdDRAnopiLNtJvPpQACtkgAzQ0l2nU/edit#gid=1304736070)):

![Graph showing effect of varying the ratio of true:false examples in training data](https://docs.google.com/spreadsheets/d/e/2PACX-1vSXwJIn0X6AcgyRXjztNHnkImcbX-EgGmLbMinxy9o69WP0D03A_3BMF3_WjJ9i42f5Zp8nz9Ypl4Ld/pubchart?oid=475854430&format=image)

I ended up choosing 1:2 - i.e. two random negative examples from different classes presented during training, for each positive one. I wanted to be in principle operating on “minimal amounts of data” and the difference between 1:2 and 1:3 seemed small.

Here’s how a one-class-per-network approach performed (each network trained for a single epoch, ([data](https://docs.google.com/spreadsheets/d/1-fzEgBgxJqEy1jdDRAnopiLNtJvPpQACtkgAzQ0l2nU/edit#gid=518897029))):

![Graph showing impact of dataset size on accuracy](https://docs.google.com/spreadsheets/d/e/2PACX-1vSXwJIn0X6AcgyRXjztNHnkImcbX-EgGmLbMinxy9o69WP0D03A_3BMF3_WjJ9i42f5Zp8nz9Ypl4Ld/pubchart?oid=1828566477&format=image)

And then, to answer the question, I compared 1-class-per-network to all-networks-all-classes ([data](https://docs.google.com/spreadsheets/d/1-fzEgBgxJqEy1jdDRAnopiLNtJvPpQACtkgAzQ0l2nU/edit#gid=1386878073)):

Naively, a network trained to classify all classes performed better. But consider the dataset sizes: each all-classes network is trained on 10,000 examples (of all classes), but each per-class network of d=10000 is trained on 1/10 as much data. So a fair comparison is between the d=10000 per-class network and d=1000 all-class network, where **per-class networks have the edge**.

Here's the result of well-trained ensembles ([data](https://docs.google.com/spreadsheets/d/1-fzEgBgxJqEy1jdDRAnopiLNtJvPpQACtkgAzQ0l2nU/edit#gid=570678490)):

![Graph showing impact of dataset size on well trained ensembles](https://docs.google.com/spreadsheets/d/e/2PACX-1vSXwJIn0X6AcgyRXjztNHnkImcbX-EgGmLbMinxy9o69WP0D03A_3BMF3_WjJ9i42f5Zp8nz9Ypl4Ld/pubchart?oid=967236200&format=image)

This was a red flag for ensembles generally: repeated re-presentation of the same data across multiple epochs reached peak performance very fast. **When the network was well trained, using an ensemble didn't have any noticeable effect**  Expanding on the far left of that graph, you can see that in the slowest case (20 examples per dataset) **ensembles larger than 50 networks had little effect, but smaller ones did perform better**:

![Graph showing impact of dataset size on well trained ensembles, for small ensemble sizes](https://docs.google.com/spreadsheets/d/e/2PACX-1vSXwJIn0X6AcgyRXjztNHnkImcbX-EgGmLbMinxy9o69WP0D03A_3BMF3_WjJ9i42f5Zp8nz9Ypl4Ld/pubchart?oid=869875220&format=image)

Conclusions here: individual networks are better homogenous; having different networks optimize to classify different classes didn’t pan out (at least in the naive way I did it).

## How could we optimize the training of an ensemble?

A friend suggested I experiment with the learning rate for small datasets - reasoning that we want individual networks to converge as quickly as possible, even if imperfectly, and rely on voting to smooth out the differences. The default learning rate in MXNet was 0.02; I compared this to 0.1, 0.2, 0.3, and 0.4, all for networks shown few examples and given a single epoch of training.

Finally, I wondered how performance changed during training: imagine a scenario where each network is trained on a single extra example (+ negatives) and then the ensemble is tested. How does performance of the ensemble change as the number of examples grows? This might be a good approximation for an ensemble that learns very slowly and naturally in-the-wild, in line with the kinds of biological plausibility that originally interested me.

### Results

A learning rate of 0.2 seemed to give the best results ([data](https://docs.google.com/spreadsheets/d/1-fzEgBgxJqEy1jdDRAnopiLNtJvPpQACtkgAzQ0l2nU/edit#gid=1846835081)):

![Graph showing impact of learning rate on ensemble performance](https://docs.google.com/spreadsheets/d/e/2PACX-1vSXwJIn0X6AcgyRXjztNHnkImcbX-EgGmLbMinxy9o69WP0D03A_3BMF3_WjJ9i42f5Zp8nz9Ypl4Ld/pubchart?oid=515576313&format=image)

And here's how that gradual learning worked out, as each ensemble size saw more examples ([data](https://docs.google.com/spreadsheets/d/1-fzEgBgxJqEy1jdDRAnopiLNtJvPpQACtkgAzQ0l2nU/edit#gid=565901898)):
/
![Graph showing impact of dataset growth on ensemble performance](https://docs.google.com/spreadsheets/d/e/2PACX-1vSXwJIn0X6AcgyRXjztNHnkImcbX-EgGmLbMinxy9o69WP0D03A_3BMF3_WjJ9i42f5Zp8nz9Ypl4Ld/pubchart?oid=903076968&format=image)

# Pulling it all together

Phew. This all took a surprising amount of time to actually run; I was doing it all on my Mac laptop, after finding that Colab instances would time out before the runs completed - some of them took a week to get through.

My conclusions from all this:

1. **Large ensembles didn’t seem useful for getting to high accuracy classifications**. Nothing I did got near to the 0.98 accuracy that this MXNet example could get to, well trained.
2. **They did compensate for a dearth of training data and/or training time, to some degree**. Getting to 0.75 accuracy with just 100 examples of each digit, just by doing it lots of times and voting, seemed... useful in theory. In practice I'm struggling to think of situations where you it'd be easier to run 1000 ensembles than iterate over the training data a network has already seen.

In retrospect this might be explained as follows: a network is initialized with random weights, training it with a few examples would bias a set of these weights towards some features in the examples, but a slightly different set each time because of the randomness of the starting position. Thus across many networks you’d end up slightly biasing towards different aspects of the training data, and thus be able, in aggregate, to classify better.

Things I didn't quite get to try:

1. Different voting schemes: I was super-naive throughout, and in particular wonder if I could derive some idea of confidence from different networks in an ensemble, just pick the confident ones?
2. MNist is a useful toy example, but I wonder if these results would replicate for other problems.