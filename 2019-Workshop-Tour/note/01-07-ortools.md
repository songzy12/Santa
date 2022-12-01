https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/124215

>  I can get better scores after I switch to use SCIP instead of CBC in ortools.
>
>  https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/122918
>
>  > Using these `100 x 176 x 176 = 3,097,600` binary variables and the right constraints, the Accounting Penalty can be linearized and fed into an MIP solver. The number of variables is huge, but the function is linear.

https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/123274

> Our cplex model proves optimality when seeded with the optimal solution in less than 3 hours with 20 thread on 2.6GHz CPUs.

> Using Gurboi, the 3M version found a solution of 69359.97 in about 6 hours and found a better solution of ~69100 after 30 hours. I let it run for 3-4 more days and nothing improved. My conclusion was that the convergence rate is not good enough to find the optimal solution in a reasonable time using the 3M version.

> We used a subset to reach the optimum, but we used all of them to prove optimality. And that used less than 16 GB of memory.

https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/122660

> 1. If the optimal solution value is known to be less than 30000, then you know for sure that `z = 0`, so you can eliminate it from the formulation (or just set it to zero, if that's easier).
> 2. Too much accuracy in the coefficients can make the calculations slower, and perhaps all those decimal places are not necessary. If you solve the problem with, say, one decimal place.

Reference paper:

https://depositonce.tu-berlin.de/bitstream/11303/1931/2/Dokument_41.pdf

