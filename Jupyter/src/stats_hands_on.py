

lst = [9,3,7,2,7,10,23,44,12,42,19,11,22,5,3,4,3,21,3]

def mean(datalist):
    total = 0
    m = 0
    for item in datalist:
        total += item
    m = total / float(len(datalist))
    return m

# result = mean(lst)
# print("Mean : {}".format(result))

def median(datalist):
    n = len(datalist)
    numsort = sorted(datalist)
    mid = n // 2
    m = 1
    if n % 2 == 0:
        lo = mid - 1
        hi = mid
        m = (numsort[lo] + numsort[hi])/2
    else:
        m = numsort[mid]
    return m

# result = median(lst)
# print("Median : {}".format(result))

def frequency_distribution(datalist):
    freqs = dict()
    for item in datalist:
        if item not in freqs.keys():
            freqs[item] = 1
        else:
            freqs[item] += 1
    return freqs

def mode(datalist):
    d = frequency_distribution(datalist)
    print(d)
    most_often = 0
    m = 0
    for item in d.keys():
        if d[item] > most_often:
            most_often = d[item]
            m = item
    return m

# result = mode(lst)
# print("Mode : {}".format(result))


from collections import Counter
def mode2(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]  # multiple modes are possible

# result = mode2(lst)
# print("Mode : {}".format(result))

def my_range(abclist):
    smallest = abclist[0]
    largest = abclist[0]
    range_of_values = 0
    for item in abclist[1:]:
        if item < smallest:
            smallest = item
        elif item > largest:
            largest = item
    range_of_values = largest - smallest
    return smallest, largest, range_of_values

# min,max,diff = my_range(lst)
# print("Range: Min {}, Max {}, Diff {}".format(min,max,diff))

def my_range2(x):
    return max(x) - min(x)

# diff = my_range2(lst)
# print("Range: {}".format(diff))

def quantile(datalist,num):
    index = int(num * len(datalist)) # slicing parameter
    return sorted(datalist)[index]
    # For values :
    # if num > .5:
    #     return sorted(datalist)[index:]
    # else:
    #     return sorted(datalist)[:index]

def interquartile_range(x):
    return	quantile(x, 0.75) - quantile(x, 0.25)

# result1 = quantile(lst,0.10)
# result2 = quantile(lst,0.25)
# result3 = quantile(lst,0.75)
# result4 = quantile(lst,0.90)
# result5 = interquartile_range(lst)
# print("Q10 {}, Q25 {}, Q50 {} Q90 {} IQR {}".format(result1,result2,result3,result4,result5))

def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def sum_of_squares(diffs):
    ss = 0
    for df in diffs:
        ss += (df) ** 2
    return ss

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def std_dev(anotherlist):
    std_dev = variance(anotherlist) ** 0.5
    return std_dev

# result1 = variance(lst)
# result2 = std_dev(lst)
# print("Variance {}, Std Dev {}".format(result1,result2))

def elementwise_multiplication(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def covariance(x, y):
    n = len(x)
    return	elementwise_multiplication(de_mean(x), de_mean(y)) / (n - 1)

def correlation(x, y):
    stdev_x = std_dev(x)
    stdev_y = std_dev(y)
    if stdev_x > 0 and stdev_y > 0:
        return	covariance(x, y) / stdev_x / stdev_y
    else:
        return	0 # if	no variation, correlation is zero

x = [2, 3, 0, 1, 3]
y = [ 2, 1, 0, 1, 2]
# result1 = covariance(x,y)
# result2 = correlation(x,y)
# print("CoVariance {}, Correlation {}".format(result1,result2))

import math
import matplotlib.pyplot as plt
def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2*math.pi)
    return	(math.exp(-(x-mu)**2/2/sigma**2) / (sqrt_two_pi* sigma))

# xs=[x	/10.0 for x in range(-50, 50)]
# plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
# plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
# plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
# plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
# plt.legend()
# plt.title("Normal pdfs")
# plt.show()