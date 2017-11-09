from timeit import timeit, Timer

int_list = list(range(10000))
str_list = [str(elem) for elem in int_list]

my_test_dict = dict(zip(str_list, int_list))


def func1(key):
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
    example for accessing dictionary using the built in 'keys' method and flush everything into a list
    """
    return key in list(my_test_dict.keys())


"""
function 1 needed: 0.01816476100066211
function 1 needed: 0.015407175000291318 (key not in dict)
function 2 needed: 0.01919510400330182
function 2 needed: 0.018761463987175375 (key not in dict)
function 3 needed: 0.020547485997667536
function 3 needed: 0.020152508004684933 (key not in dict)
function 4 needed: 5.34116512699984
function 4 needed: 14.085671805005404 (key not in dict)

"""

t1 = Timer(lambda: func1('1'))

time1 = t1.timeit(number=100000)

print(f"function 1 needed: {time1}")

t1 = Timer(lambda: func1('100001'))

time1 = t1.timeit(number=100000)

print(f"function 1 needed: {time1} (key not in dict)")

t2 = Timer(lambda: func2('1'))

time2 = t2.timeit(number=100000)

print(f"function 2 needed: {time2}")

t2 = Timer(lambda: func2('100001'))

time2 = t2.timeit(number=100000)

print(f"function 2 needed: {time2} (key not in dict)")

t3 = Timer(lambda: func3('1'))

time2 = t3.timeit(number=100000)

print(f"function 3 needed: {time2}")

t3 = Timer(lambda: func3('100001'))

time2 = t3.timeit(number=100000)

print(f"function 3 needed: {time2} (key not in dict)")

t4 = Timer(lambda: func4('1'))

time2 = t4.timeit(number=100000)

print(f"function 4 needed: {time2}")

t4 = Timer(lambda: func4('100001'))

time2 = t4.timeit(number=100000)

print(f"function 4 needed: {time2} (key not in dict)")
