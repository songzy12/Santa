import pandas as pd

N_DAYS = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125


def cost_function(prediction):
    penalty = 0
    days = list(range(N_DAYS, 0, -1))
    tmp = pd.read_csv(
        "../input/santa-workshop-tour-2019/family_data.csv", index_col="family_id"
    )
    family_size_dict = tmp[["n_people"]].to_dict()["n_people"]

    cols = [f"choice_{i}" for i in range(10)]
    choice_dict = tmp[cols].to_dict()

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k: 0 for k in days}

    # Looping over each family; d is the day for each family f
    for f, d in enumerate(prediction):
        # Using our lookup dictionaries to make simpler variable names
        n = family_size_dict[f]
        choice_0 = choice_dict["choice_0"][f]
        choice_1 = choice_dict["choice_1"][f]
        choice_2 = choice_dict["choice_2"][f]
        choice_3 = choice_dict["choice_3"][f]
        choice_4 = choice_dict["choice_4"][f]
        choice_5 = choice_dict["choice_5"][f]
        choice_6 = choice_dict["choice_6"][f]
        choice_7 = choice_dict["choice_7"][f]
        choice_8 = choice_dict["choice_8"][f]
        choice_9 = choice_dict["choice_9"][f]

        # add the family member count to the daily occupancy
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d == choice_0:
            penalty += 0
        elif d == choice_1:
            penalty += 50
        elif d == choice_2:
            penalty += 50 + 9 * n
        elif d == choice_3:
            penalty += 100 + 9 * n
        elif d == choice_4:
            penalty += 200 + 9 * n
        elif d == choice_5:
            penalty += 200 + 18 * n
        elif d == choice_6:
            penalty += 300 + 18 * n
        elif d == choice_7:
            penalty += 300 + 36 * n
        elif d == choice_8:
            penalty += 400 + 36 * n
        elif d == choice_9:
            penalty += 500 + 36 * n + 199 * n
        else:
            penalty += 500 + 36 * n + 398 * n

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for _, v in daily_occupancy.items():
        if v < MIN_OCCUPANCY:  # (v > MAX_OCCUPANCY) or
            penalty += 100000000

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (
        (daily_occupancy[days[0]] - 125.0) / 400.0 * daily_occupancy[days[0]] ** (0.5)
    )
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_costs = [max(0, accounting_cost)]
    diffs = [0]
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_costs.append(
            max(0, (today_count - 125.0) / 400.0 * today_count ** (0.5 + diff / 50.0))
        )
        yesterday_count = today_count

    return penalty, sum(accounting_costs), penalty + sum(accounting_costs)


NUMBER_DAYS = 100
NUMBER_FAMILIES = 5000
COST_PER_FAMILY = [0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500]
COST_PER_FAMILY_MEMBER = [0, 0, 9, 9, 9, 18, 18, 36, 36, 235, 434]

# preference
data = pd.read_csv("../input/santa-workshop-tour-2019/family_data.csv")
columns = data.columns[1:11]
DESIRED = data[columns].values
N_PEOPLE = data["n_people"].values
print("preference data read")

# assignment
submission = pd.read_csv("../input/santa-workshop-tour-2019/sample_submission.csv")
assigned_days = submission["assigned_day"].values
print("assignment data read")

# init solver
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver(
    "Optimization preference cost", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING
)

# assign[i][j],
# i in range(5000), j in range(100):
# assign family i to day j
# this would induce the preference cost in object

PCOSTM, B = {}, {}
for fid in range(NUMBER_FAMILIES):
    for i in range(NUMBER_DAYS):
        B[fid, i] = solver.BoolVar("")
        PCOSTM[fid, i] = (
            COST_PER_FAMILY[-1] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[-1]
        )
    for i in range(10):
        PCOSTM[fid, DESIRED[fid][i] - 1] = (
            COST_PER_FAMILY[i] + N_PEOPLE[fid] * COST_PER_FAMILY_MEMBER[i]
        )
print("B initialized")

# constraint:
# sum_j assign[i][j] = 1
# sum_i assign[i][j] \in [125, 300]

for fid in range(NUMBER_FAMILIES):
    solver.Add(solver.Sum([B[fid, j] for j in range(NUMBER_DAYS)]) == 1)

for day in range(NUMBER_DAYS):
    solver.Add(
        solver.Sum(
            [
                B[fid, day] * N_PEOPLE[fid] for fid in range(NUMBER_FAMILIES)
            ]
        )
        >= 125
    )
    solver.Add(
        solver.Sum(
            [
                B[fid, day] * N_PEOPLE[fid] for fid in range(NUMBER_FAMILIES)
            ]
        )
        <= 300
    )
print("B constraints added")

# day[i][j],
# i in range(100), j in range(125, 300):
# number of families at day i equals j

daily_occupancy_ = {}
for i in range(NUMBER_DAYS):
    for j in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
        daily_occupancy_[i, j] = solver.BoolVar("")
print("daily occupancy initialized")

# constraint:
# sum_j day[i][j] = 1
# day[i][j] = 1 if and only if sum_k assign[k][i] = j
# day[i][j] * j = sum_k assign[k][i]

for i in range(NUMBER_DAYS):
    solver.Add(
        solver.Sum(
            [daily_occupancy_[i, j] for j in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)]
        )
        == 1
    )
for i in range(NUMBER_DAYS):
    solver.Add(
        solver.Sum(
            [
                daily_occupancy_[i, j] * j
                for j in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
            ]
        )
        == solver.Sum(
            [B[k, i] for k in range(NUMBER_FAMILIES)]
        )
    )
print("daily occupancy constraints added")

# account[i][j][k],
# i in range(100), j in range(125, 300), k in range(125, 300)
# day[i] equals j and day[i+1] equals k
# this would induce the accouting penalty in object

account = {}

optimal = 68888.04

cost_dp = {}


def compute_accounting_cost(today_count, yesterday_count):
    if (today_count, yesterday_count) not in cost_dp:
        diff = abs(today_count - yesterday_count)
        cost = max(
            0, (today_count - 125.0) / 400.0 * today_count ** (0.5 + diff / 50.0)
        )
        cost_dp[today_count, yesterday_count] = int(cost * 10) / 10
    return cost_dp[today_count, yesterday_count]


# constraint:
# account[i][j][k] = 1 if and only if day[i][j] = 1 and day[i+1][k] = 1
# account[i][j][k] = day[i][j] & day[i+1][k]

# for y = x1 & x2:
# y \ge x1 + x2 - 1
# y \le x1
# y \le x2


for i in range(NUMBER_DAYS - 1):
    for j in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
        for k in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1):
            account[i, j, k] = solver.BoolVar("")
            solver.Add(
                account[i, j, k]
                >= daily_occupancy_[i, j] + daily_occupancy_[i + 1, k] - 1
            )
            solver.Add(account[i, j, k] <= daily_occupancy_[i, j])
            solver.Add(account[i, j, k] <= daily_occupancy_[i + 1, k])
print("account initialized and constraints added")

# object
solver.Minimize(
    solver.Sum([PCOSTM[fid, day] * B[fid, day] for fid, day in B])
    + solver.Sum(
        [
            solver.Sum(
                [
                    solver.Sum(
                        [
                            account[i, j, k] * compute_accounting_cost(j, k)
                            for i in range(NUMBER_DAYS - 1)
                        ]
                    )
                    for j in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
                ]
            )
            for k in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
        ]
    )
    + solver.Sum(
        [
            daily_occupancy_[NUMBER_DAYS - 1, k] * compute_accounting_cost(k, k)
            for k in range(MIN_OCCUPANCY, MAX_OCCUPANCY + 1)
        ]
    )
)
print("objective function initialized")

# meta data of the solver
# https://www.kaggle.com/docs/kernels#technical-specifications
# 9 hours execution time
# 4 CPU cores, 16 Gigabytes of RAM
NUM_SECONDS = 3600
NUM_THREADS = 1
solver.set_time_limit(NUM_SECONDS * NUM_THREADS * 1000)
solver.SetNumThreads(NUM_THREADS)
print("solver set")

# solve
sol = solver.Solve()

# post process
status = [
    "OPTIMAL",
    "FEASIBLE",
    "INFEASIBLE",
    "UNBOUNDED",
    "ABNORMAL",
    "MODEL_INVALID",
    "NOT_SOLVED",
]
if status[sol] in ["OPTIMAL", "FEASIBLE"]:
    tmp = assigned_days.copy()
    for fid, day in B:
        if B[fid, day].solution_value() > 0.5:
            tmp[fid] = day + 1

    assigned_days = tmp
    submission["assigned_day"] = assigned_days
    submission.to_csv("submission.csv", index=False)
    print("Result:", status[sol], cost_function(tmp))
else:
    print("Result:", status[sol])

