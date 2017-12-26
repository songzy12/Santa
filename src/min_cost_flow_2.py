#FORK FROM https://www.kaggle.com/zfturbo/happiness-vs-gift-popularity-v2-0-89?scriptVersionId=2008088
# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd
import numpy as np
from collections import Counter
import operator
import math

INPUT_PATH = './input/'

def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    # in case of large numbers, using floor division
    return a * b // math.gcd(a, b)

def avg_normalized_happiness(pred, gift, wish):
    
    n_children = 1000000 # n children to give
    n_gift_type = 1000 # n types of gifts available
    n_gift_quantity = 1000 # each type of gifts are limited to this quantity
    n_gift_pref = 100 # number of gifts a child ranks
    n_child_pref = 1000 # number of children a gift ranks
    twins = math.ceil(0.04 * n_children / 2.) * 2    # 4% of all population, rounded to the closest number
    triplets = math.ceil(0.005 * n_children / 3.) * 3    # 0.5% of all population, rounded to the closest number
    ratio_gift_happiness = 2
    ratio_child_happiness = 2

    # check if triplets have the same gift
    for t1 in np.arange(0, triplets, 3):
        triplet1 = pred[t1]
        triplet2 = pred[t1+1]
        triplet3 = pred[t1+2]
        # print(t1, triplet1, triplet2, triplet3)
        assert triplet1 == triplet2 and triplet2 == triplet3
                
    # check if twins have the same gift
    for t1 in np.arange(triplets, triplets+twins, 2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        # print(t1)
        assert twin1 == twin2

    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)
    
    for i in range(len(pred)):
        child_id = i
        gift_id = pred[i]
        
        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0 
        assert gift_id >= 0
        child_happiness = (n_gift_pref - np.where(wish[child_id]==gift_id)[0]) * ratio_child_happiness
        if not child_happiness:
            child_happiness = -1

        gift_happiness = ( n_child_pref - np.where(gift[gift_id]==child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness
        
    denominator1 = n_children*max_child_happiness
    denominator2 = n_gift_quantity*max_gift_happiness*n_gift_type
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    ret = float(math.pow(total_child_happiness*multiplier,3) + \
        math.pow(np.sum(total_gift_happiness),3)) / float(math.pow(common_denom,3))
    return ret
    

def get_overall_hapiness(wish, gift):


    res_child = dict()
    for i in range(0, wish.shape[0]):
        for j in range(56):
            res_child[(i, wish[i][j])] = (1 + (wish.shape[1] - j)*2) / 20

    res_santa = dict()
    for i in range(gift.shape[0]):
        for j in range(gift.shape[1]):
            res_santa[(gift[i][j], i)] = (1 + (gift.shape[1] - j)*2) / 2000

    positive_cases = list(set(res_santa.keys()) | set(res_child.keys()))
    print('Positive case tuples (child, gift): {}'.format(len(positive_cases)))

    res = dict()
    for p in positive_cases:
        res[p] = 0
        if p in res_child:
            res[p] += res_child[p]
        if p in res_santa:
            res[p] += res_santa[p]
    return res


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def get_most_desired_gifts(wish, gift):
    best_gifts = value_counts_for_list(np.ravel(wish))
    return best_gifts


def recalc_hapiness(happiness, best_gifts, gift):
    recalc = dict()
    for b in best_gifts:
        recalc[b[0]] = b[1] / 1000

    for h in happiness:
        c, g = h
        happiness[h] /= recalc[g]

        # Make twins more happy
        if c <= 45000 and happiness[h] < 0.01:
            happiness[h] = 0.01

    return happiness


def solve():
    wish = pd.read_csv(INPUT_PATH + 'child_wishlist_v2.csv', header=None).as_matrix()[:, 1:]
    gift_init = pd.read_csv(INPUT_PATH + 'gift_goodkids_v2.csv', header=None).as_matrix()[:, 1:]
    gift = gift_init.copy()
    answ = np.zeros(len(wish), dtype=np.int32)
    answ[:] = -1
    gift_count = np.zeros(len(gift), dtype=np.int32)

    happiness = get_overall_hapiness(wish, gift)
    best_gifts = get_most_desired_gifts(wish, gift)
    happiness = recalc_hapiness(happiness, best_gifts, gift)
    sorted_hapiness = sort_dict_by_values(happiness)
    print('Happiness sorted...')

    for i in range(len(sorted_hapiness)):
        child = sorted_hapiness[i][0][0]
        g = sorted_hapiness[i][0][1]
        if answ[child] != -1:
            continue
        if gift_count[g] >= 1000:
            continue
        if child <= 5000 and gift_count[g] < 997:
            if child % 3 == 0:
                answ[child] = g
                answ[child+1] = g
                answ[child+2] = g
                gift_count[g] += 3
            elif child % 3 == 1:
                answ[child] = g
                answ[child-1] = g
                answ[child+1] = g
                gift_count[g] += 3
            else:
                answ[child] = g
                answ[child-1] = g
                answ[child-2] = g
                gift_count[g] += 3
        elif child > 5000 and child <= 45000 and gift_count[g] < 998:
            if child % 2 == 0:
                answ[child] = g
                answ[child - 1] = g
                gift_count[g] += 2
            else:
                answ[child] = g
                answ[child + 1] = g
                gift_count[g] += 2
        elif child > 45000:
            answ[child] = g
            gift_count[g] += 1

    print('Left unhappy:', len(answ[answ == -1]))
    
    # unhappy children
    for child in range(45001, len(answ)):
        if answ[child] == -1:
            g = np.argmin(gift_count)
            answ[child] = g
            gift_count[g] += 1

    if answ.min() == -1:
        print('Some children without present')
        exit()

    if gift_count.max() > 1000:
        print('Some error in kernel: {}'.format(gift_count.max()))
        exit()

    print('Start score calculation...')
    # score = avg_normalized_happiness(answ, gift_init, wish)
    # print('Predicted score: {:.8f}'.format(score))
    score = avg_normalized_happiness(answ, gift, wish)
    print('Predicted score: {:.8f}'.format(score))

    out = open('01_public_subm.csv', 'w')
    out.write('ChildId,GiftId\n')
    for i in range(len(answ)):
        out.write(str(i) + ',' + str(answ[i]) + '\n')
    out.close()

solve()

#THE1OWL
from sklearn.utils import shuffle
from collections import Counter
from multiprocessing import *
from random import randint
import pandas as pd
import numpy as np
import copy, random
import math

gp = pd.read_csv('../input/child_wishlist_v2.csv',header=None).drop(0, 1).values
cp = pd.read_csv('../input/gift_goodkids_v2.csv',header=None).drop(0, 1).values

def ANH_SCORE(pred):
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= 1000

    for t1 in np.arange(0,5001,3):
        triplet1 = pred[t1]
        triplet2 = pred[t1+1]
        triplet3 = pred[t1+2]
        assert triplet1[1] == triplet2[1] and triplet2[1] == triplet3[1]
    
    for t1 in np.arange(5001,45001, 2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        assert twin1[1] == twin2[1]

    tch = 0
    tgh = np.zeros(1000)
    
    for row in pred:
        cid, gid = row

        assert cid < 1e6
        assert gid < 1000
        assert cid >= 0 
        assert gid >= 0
        
        ch = (100 - np.where(gp[cid]==gid)[0]) * 2
        if not ch:
            ch = -1

        gh = (1000 - np.where(cp[gid]==cid)[0]) * 2
        if not gh:
            gh = -1

        tch += ch
        tgh[gid] += gh
    return float(math.pow(tch*10,3) + math.pow(np.sum(tgh),3)) / 8e+27

#print(ANH_SCORE(test))

def ANH_SCORE_ROW(pred):
    tch = 0
    tgh = np.zeros(1000)
    for row in pred:
        cid, gid = row
        ch = (100 - np.where(gp[cid]==gid)[0]) * 2
        if not ch:
            ch = -1
        gh = (1000 - np.where(cp[gid]==cid)[0]) * 2
        if not gh:
            gh = -1
        tch += ch
        tgh[gid] += gh
    return float(math.pow(tch*10,3) + math.pow(np.sum(tgh),3)) / 8e+27 #math.pow(float(tch)/2e8,2) + math.pow(np.mean(tgh)/2e6,2)

def metric_function(c1, c2):
    cid1, gid1 = c1
    cid2, gid2 = c2
    return [ANH_SCORE_ROW([c1,c2]), ANH_SCORE_ROW([[cid1,gid2],[cid2,gid1]])]

def objective_function_swap(otest):
    otest = otest.values
    otest = shuffle(otest, random_state=2017)
    #score1 = ANH_SCORE_ROW(otest)
    for b in range(len(otest)):
        for j in range(b+1,len(otest)):
            mf = metric_function(otest[b], otest[j])
            if mf[0] < mf[1]:
                temp = int(otest[b][1])
                otest[b][1] = int(otest[j][1])
                otest[j][1] = temp
                break
    #score2 = ANH_SCORE_ROW(otest)
    #if score2 > score1:
        #print(score2 - score1)
    otest = pd.DataFrame(otest)
    return otest

def multi_transform(mtest):
    p = Pool(cpu_count())
    mtest = p.map(objective_function_swap, np.array_split(mtest, cpu_count()*30))
    mtest = pd.concat(mtest, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    return mtest

if __name__ == '__main__':
    test = pd.read_csv('01_public_subm.csv')
    test2 = multi_transform(shuffle(test[45001:100000].copy(), random_state=2017))
    test = pd.concat([pd.DataFrame(test[:45001].values), pd.DataFrame(test2), pd.DataFrame(test[100000:].values)], axis=0, ignore_index=True).reset_index(drop=True).values
    test = pd.DataFrame(test)
    test.columns = ['ChildId','GiftId']
    print(ANH_SCORE(test.values))
    test.to_csv('02_public_subm.csv', index=False)
