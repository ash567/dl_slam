import random

val = range(11)
random.seed(1)
random.shuffle(val)
print val[:5]
