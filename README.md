# Lottery-Ticket-Hypothesis
`A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations,` - Hypothesis

## Algorithmic Description:
### Identifying winning tickets. 
To identify a winning ticket, train a network and prune its smallest-magnitude weights. The remaining, unpruned connections constitute the architecture of the winning ticket. Unique to this work, each unpruned connection’s value is then reset to its initialization from original network before it was trained. This forms the central experiment:

1. Randomly initialize a neural network f(x; θ0) (where θ0 ∼ Dθ).
2. Train the network for j iterations, arriving at parameters θj .
3. Prune p% of the parameters in θj , creating a mask m.
4. Reset the remaining parameters to their values in θ0, creating the winning ticket f(x; mθ0).

As described, this pruning approach is one-shot: the network is trained once, p% of weights are pruned, and the surviving weights are reset. However, in this paper, the focus is on iterative pruning, which repeatedly trains, prunes, and resets the network over n rounds; each round prunes p^(1/n)% of the weights that survive the previous round. The results show that iterative pruning finds winning tickets that match the accuracy of the original network at smaller sizes than does one-shot pruning.
