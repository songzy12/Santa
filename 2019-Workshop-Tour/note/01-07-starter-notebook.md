## starter notebook

https://www.kaggle.com/inversion/santa-s-2019-starter-notebook

### family_data

|| choice_0  | choice_1 | choice_2 | choice_3 | choice_4 | choice_5 | choice_6 | choice_7 | choice_8 | choice_9 | n_people |
| :-------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :--- |
| family_id |          |          |          |          |          |          |          |          |          |          |      |
| 0         | 52       | 38       | 12       | 82       | 33       | 75       | 64       | 76       | 10       | 28       | 4    |
| 1         | 26       | 4        | 82       | 5        | 11       | 47       | 38       | 6        | 66       | 61       | 4    |
| 2         | 100      | 54       | 25       | 12       | 27       | 82       | 10       | 89       | 80       | 33       | 3    |
| 3         | 2        | 95       | 1        | 96       | 32       | 6        | 40       | 31       | 9        | 59       | 2    |
| 4         | 53       | 1        | 47       | 93       | 26       | 3        | 46       | 16       | 42       | 39       | 4    |

### sample_submission

|| assigned_day |
| :----------- | :--- |
| family_id    |      |
| 0            | 100  |
| 1            | 99   |
| 2            | 98   |
| 3            | 97   |
| 4            | 96   |

### cost

https://www.kaggle.com/c/santa-workshop-tour-2019/overview/evaluation

1. The total number of *people* attending the workshop each day must be between **125 - 300**
2. Santa provides consolation gifts (of varying value) to families according to their assigned day relative to their preferences.

- `choice_0`: *no consolation gifts*
- `choice_1`: one **$50** gift card to Santa's Gift Shop
- `choice_2`: one **$50** gift card, and 25% off Santa's Buffet (value **$9**) for each family member

- `choice_3`: one **$100** gift card, and 25% off Santa's Buffet (value **$9**) for each family member

- `choice_4`: one **$200** gift card, and 25% off Santa's Buffet (value **$9**) for each family member

- `choice_5`: one **$200** gift card, and 50% off Santa's Buffet (value **$18**) for each family member

- `choice_6`: one **$300** gift card, and 50% off Santa's Buffet (value **$18**) for each family member

- `choice_7`: one **$300** gift card, and free Santa's Buffet (value **$36**) for each family member

- `choice_8`: one **$400** gift card, and free Santa's Buffet (value **$36**) for each family member

- `choice_9`: one **$500** gift card, and free Santa's Buffet (value **$36**) for each family member, and 50% off North Pole Helicopter Ride tickets (value **$199**) for each family member

- `otherwise`: one **$500** gift card, and free Santa's Buffet (value **$36**) for each family member, and free North Pole Helicopter Ride tickets (value **$398**) for each family member

3. $ accounting\: penalty = \sum_{d=100}^{1} \frac{(N_{d} - 125)}{400} {N_d}^{( \frac{1}{2} + \frac{\lvert N_d - N_{d+1} \rvert }{50} )} $

## python

```
N_DAYS = 100

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in days}

    # Looping over each family; d is the day for each family f
    for f, d in enumerate(prediction):    
        # Using our lookup dictionaries to make simpler variable names
        n = family_size_dict[f]
        
        # add the family member count to the daily occupancy
        daily_occupancy[d] += n
```

```
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
```

```
    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[days[0]]-125.0) / 400.0 * daily_occupancy[days[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[days[0]]
    for day in days[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count
```

```
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125
    
    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    for _, v in daily_occupancy.items():
        if (v > MAX_OCCUPANCY) or (v < MIN_OCCUPANCY):
            penalty += 100000000
```

