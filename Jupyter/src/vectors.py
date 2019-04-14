vv = [ 1, 2,3]
ww = [3,2,1]

def vector_add(v, w):
    """adds corresponding elements"""
    return [v_i + w_i
            for v_i, w_i in zip(v, w)]

# result = vector_add(vv,ww)
# print("Vector Addition {}".format(result))

def vector_subtract(v, w):
    """subtracts corresponding elements"""
    return [v_i - w_i
            for v_i, w_i in zip(v, w)]

# result = vector_subtract(vv,ww)
# print("Vector Subtraction {}".format(result))

def vector_sum(vectors):
    """sums all corresponding elements"""
    result = vectors[0]                         # start with the first vector
    for vector in vectors[1:]:                  # then loop over the others
        result = vector_add(result, vector)     # and add them to the result
    return result

vecs = [[ 1, 2,3],[3,2,1],[3,2,-1]]
# result = vector_sum(vecs)
# print("Vectors Sum {}".format(result))

def scalar_multiply(c, v):
    """c is a number, v is a vector"""
    return [c * v_i for v_i in v]

cc = 4
# result =  scalar_multiply(cc,vv)
# print("Scalar Multiply{}".format(result))

def vector_mean(vectors):
    """compute the vector whose ith element is the mean of the ith elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

# result =  vector_mean(vecs)
# print("Vectors Mean {}".format(result))

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i
               for v_i, w_i in zip(v, w))

# result =  dot(vv,ww)
# print("Dot Product {}".format(result))

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

import math
def magnitude(v):
    return math.sqrt(sum_of_squares(v))   # math.sqrt is square root function

def squared_distance(v, w):
    """(v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(vector_subtract(v, w))

result =  squared_distance(vv,ww)
print("Squared Distance {}".format(result))


def distance(v, w):
   return math.sqrt(squared_distance(v, w))

result =  distance(vv,ww)
print("Distance {}".format(result))

def distance2(v, w):
    return magnitude(vector_subtract(v, w))

result =  distance2(vv,ww)
print("Distance2 {}".format(result))

vv = [ 1, 2,3]
ww = [3,2,1]
vecs = [[ 1, 2,3],[3,2,1],[3,2,-1]]
cc = 4
# result = vector_subtract(vv,ww)
# result = vector_sum(vecs)
# result =  scalar_multiply(cc, vv)
# result = vector_mean(vecs)
# result = dot(vv,ww)
# result = distance(vv,ww)
# result = distance2(vv,ww)
# print(result)