if __name__ == "__main__":
    from timeit import timeit, Timer
    ntimes = 100000    
    from my_module import A, c_function, dostuff
    from pure_module import B, _slow_helper, some_function

    t = Timer(lambda: A().foo(5))
    print(f"dict needed: {t.timeit(number=ntimes)/ntimes}")

    #t = Timer(lambda: dostuff(ntimes))

    #print(f"dict needed: {t.timeit(number=ntimes)/ntimes}")
    
    print("pure_python")
    
    t = Timer(lambda: B().foo(5))
    print(f"dict needed: {t.timeit(number=ntimes)/ntimes}")

    #t = Timer(lambda:_slow_helper(some_function(3)))
    #print(f"dict needed: {t.timeit(number=ntimes)/ntimes}")
