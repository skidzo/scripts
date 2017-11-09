from timeit import timeit, Timer
from collections import OrderedDict
"""
creating dict vs. OrderedDict::
    dict needed: 0.0005610200763300235
    OrderedDict needed: 0.001250969027789979 (key not in dict)

checking if key is in dict::
    function 1 needed: 1.409474600222893e-07
    function 1 needed: 1.3962903991341592e-07 (key not in dict)
    function 2 needed: 1.8527951993746682e-07
    function 2 needed: 1.8942816997878254e-07 (key not in dict)
    function 3 needed: 1.9767146004596726e-07
    function 3 needed: 2.0610659004887566e-07 (key not in dict)
    function 4 needed: 5.84014135999314e-05
    function 4 needed: 0.00014321265802005656 (key not in dict)

or OrderedDict::
    function 5 needed: 1.5774050989421086e-07
    function 5 needed: 1.5929286993923597e-07 (key not in dict)

"""

int_list = list(range(10000))
str_list = [str(elem) for elem in int_list]
ntimes = 100000


def create_dict():
    return dict(zip(str_list, int_list))


def create_ordered_dict():
    return OrderedDict(zip(str_list, int_list))


t = Timer(lambda: create_dict())
print(f"dict needed: {t.timeit(number=ntimes)/ntimes}")

t = Timer(lambda: create_ordered_dict())
print(
    f"OrderedDict needed: {t.timeit(number=ntimes)/ntimes} (key not in dict)")

my_test_dict = create_dict()
my_ordered_dict = create_ordered_dict()


def func1(key, my_test_dict=my_test_dict):
    """
    example for accessing dictionary with 'in'
    """
    return key in my_test_dict


def func2(key):
    """
    example for accessing dictionary with 'in' using the built in 'keys' method
    """
    return key in my_test_dict.keys()


def func3(key):
    """
    example for accessing dictionary using the built in 'get' method
    """
    value = my_test_dict.get(key)
    if value is None:
        return False
    else:
        return True


def func4(key):
    """
    example for accessing dictionary using the built in 'keys' method and flush 
    everything into a list

    definately always worse than using an OrderedDict!
    """
    return key in list(my_test_dict.keys())


t = Timer(lambda: func1('1'))
print(f"function 1 needed: {t.timeit(number=ntimes)/ntimes}")

t = Timer(lambda: func1('100001'))
print(f"function 1 needed: {t.timeit(number=ntimes)/ntimes} (key not in dict)")

t = Timer(lambda: func2('1'))
print(f"function 2 needed: {t.timeit(number=ntimes)/ntimes}")

t = Timer(lambda: func2('100001'))
print(f"function 2 needed: {t.timeit(number=ntimes)/ntimes} (key not in dict)")

t = Timer(lambda: func3('1'))
print(f"function 3 needed: {t.timeit(number=ntimes)/ntimes}")

t = Timer(lambda: func3('100001'))
print(f"function 3 needed: {t.timeit(number=ntimes)/ntimes} (key not in dict)")

t = Timer(lambda: func4('1'))
print(f"function 4 needed: {t.timeit(number=ntimes)/ntimes}")

t = Timer(lambda: func4('100001'))
print(f"function 4 needed: {t.timeit(number=ntimes)/ntimes} (key not in dict)")

t = Timer(lambda: func1('1', my_test_dict=my_ordered_dict))
print(f"function 5 needed: {t.timeit(number=ntimes)/ntimes}")

t = Timer(lambda: func1('100001', my_test_dict=my_ordered_dict))
print(f"function 5 needed: {t.timeit(number=ntimes)/ntimes} (key not in dict)")
