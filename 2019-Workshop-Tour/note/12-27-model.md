https://www.kaggle.com/c/santa-workshop-tour-2019/discussion/120103

## Variables

x[i, j]: whether we put family i on day j.

y[j]: the number of families on day j.

Variable 1: x[i, j], i = 1:n, j = 1:m, type = "binary"

Variable 2: y[j], j = 1:(m+1), type = "integer"

## Objective
Objective is to minimize the (preference\_cost + accounting\_penalty), 

defined as

Preference Cost = sum\_expr(preference\_cost(i, j) * x[i, j], i = 1:n, j = 1:m)

Accounting Penalty = sum\_expr(accounting\_penalty(y[j], y[j+1]) , j = 1:m)

## Constraint
Constraint 1: sum\_expr(x[i, j], j = 1:m) == 1, i = 1:n

Constraint 2: sum\_expr(x[i, j] * n_people[i], i = 1:n) == y[j], j = 1:m)

Constraint 3: y[j] <= 300 , j = 1:m

Constraint 4: y[j] >= 125 , j = 1:m

Constraint 5: y[m] == y[m+1]

## Solver

Gurobi

Cplex 

LocalSolver 

OR-Tools