https://www.kaggle.com/competitions/santa-gift-matching

## Winner Solution

https://www.kaggle.com/c/santa-gift-matching/discussion/47376

* maximize `1000000 ANCH + ANSH` rather than `ANCH^3 +ANSH^3 `.
* relaxed problem ignoring twin / triplet constraints as min cost flow problem.
* most of the twins / triplets in optimal assignment for relaxed problem follow the constraints.
* if we know how many singles, twins, triplets receive each present, then the original problem can be formulated as min-cost flow problem.

## 2nd Place Solution

https://www.kaggle.com/c/santa-gift-matching/discussion/47386



## 3rd Place Solution

https://www.kaggle.com/c/santa-gift-matching/discussion/47374

* The key idea is to maximize `a*ANCH + b*ANSH` (a >> b, a=100000, b=1, for example) instead of `ANCH^3 + ANSH^3`.
* One more problem is twins and triplets.  We maximized `a*ANCH + b*ANSH + c(a>>b>>c)` remove twins and triplets with same Gift, and repeat.
* Proof for optimal solution: https://k-harada.github.io/santa2017_18.html

## Improver

**score: 0.935401194595**

## Min Cost Flow Problem with Constraint

Kernel: <https://www.kaggle.com/zfturbo/max-flow-with-min-cost-v2-0-9267/code>

Possible Improvement: https://www.kaggle.com/c/santa-gift-matching/discussion/46559

```python
for i in range(min_cost_flow.NumArcs()):
      cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
      print('%1s -> %1s   %3s  / %3s       %3s' % (
          min_cost_flow.Tail(i),
          min_cost_flow.Head(i),
          min_cost_flow.Flow(i),
          min_cost_flow.Capacity(i),
          cost))
```

But he did not check whether `min_cost_flow.Flow(i) == min_cost_flow.Capacity(i)`. If this is the case, then gift overflow will occur. So he just simply fixed the gift overflow issue to solve this. Here the overflowed gifts are exactly due to the unfilled flows.

Gift Overflow Simple Fix: from singles find the overflowed gifts, then change them to the most left gifts. 

**score: 0.9313371252, rank: 96**

If we do some normalization in the end: 

**score: 0.9313577105, rank: 90**

If we do some normalization in the middle: 

**score: 0.894287284962**

```
if p in res_child:
	a = res_child[p]
	res[p] += int((a ** 3) * 5)
if p in res_santa:
	b = res_santa[p]
	res[p] += int((b ** 3) / 5)
```

Change `4` to be `5`:

**score:0.931497977947, rank: 103**

Change `4` to be `10`:

**score:0.931678237749**

## Min Cost Flow Problem

<https://en.wikipedia.org/wiki/Minimum-cost_flow_problem>

<https://www.kaggle.com/c/santa-gift-matching/discussion/45857>

<https://www.kaggle.com/zfturbo/max-flow-with-min-cost-in-10-minutes-0-9408/code>

<https://developers.google.com/optimization/flow/mincostflow>

### Algorithm

* construct a relaxed graph
* use or tools to solve the min cost flow problem
* exchange some gifts greedily to solve conflicts

### Experiment

* If we construct a undirected graph, where we connect all the gifts to children, with capacity being 1, and cost being overall happiness: 

  **score: 0.8831214439566526**

* If we change the cost of each edge to recalculated happiness: 

  **score: 0.8830931904105714**

* If we change the cost of each edge to average of triplets/twins: 

  **score: 0.9221225541, rank: 146**

Note that all the three methods above we have to adjust the assignments of triplets or twins after solving the min cost flow problem. The method is to find the worst child from the singles who has the same gift with the first child in the triplets or twins, then exchange his gift with the others in the triplets or twins. Here the `worst` means that he is most unsatisfied with this particular gift. Note here we will not have unassigned ones as in heuristic method.

## Heuristic

Just sort the (child, gift) pair based on recalculated happiness, then do a greedy assignment. Recalculated happiness means divide the happiness of (child, gift) pair with the number of children who wants that gift. Greedy assignment means first satisfy the most happiest pair, and so on. Then for the unassigned ones, just assign the left gifts one by one.

**score: 0.8920217322, rank: 131**

**score: 0.8921821552, rank: 101**

**score: 0.8938008609, rank: 122**

> To beginners, use 100 instead of 55 will give you 0.8938008609, however you need ~ 28GB ram if you do so



* `get_overall_happiness`:
* `get_most_desired_gifts`:
* `recalc_happiness`:
* `sort_dict_by_values`:

```python
child = sorted_hapiness[i][0][0]
g = sorted_hapiness[i][0][1]
```

## MISC

or-tools: Google's Operations Research tools

DIMACS (Center for Discrete Mathematics and Theoretical Computer Science)

SAT: SATISFIABILITY

MIP (mixed integer programming) tools: Gurobi and CPLEX

or bust: or nothing, or die trying

##Input

* 1,000,000 children,  with wish list of 100 gifts
* 1000 gifts, with list of 1000 good kids
* 0-5000: triplets
* 5001-45000: twins

## Evaluation

`Average Normalized Happiness (ANH) = (AverageNormalizedChildHappiness (ANCH) ) ^ 3 + (AverageNormalizedSantaHappiness (ANSH) ) ^ 3`



`ANCH=1 / n_c \sum_{i=0}^{n_c−1} ChildHappiness / MaxChildHappiness`

`ANSH=1 / n_g \sum_{i=0}^{n_g−1} GiftHappiness / MaxGiftHappiness`



`MaxChildHappiness = len(ChildWishList) * 2`

`MaxGiftHappiness = len(GiftGoodKidsList) * 2`

`ChildHappiness = 2 * GiftOrder` if the gift is found in the wish list of the child.

`ChildHappiness = -1` if the gift is out of the child's wish list.

`GiftHappiness = 2 * ChildOrder` if the child is found in the good kids list of the gift.

`GiftHappiness = -1` if the child is out of the gift's good kids list.
