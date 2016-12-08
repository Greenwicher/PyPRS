def dominating(p1,p2):
    """determines whether p1 dominates p2
    Args:
        p1,p2: Two double numpy arrays representing the performance averages 
               vector of this point/solutions
    Returns:
        flag: A boolean value indicating whether p1 dominates p2
    """
    maxDiff,minDiff = -np.inf,np.inf
    for i in range(len(p1)):
        diff = p1[i] - p2[i]
        if diff > maxDiff: maxDiff = diff
        if diff < minDiff: minDiff = diff
    flag = maxDiff<=0 and minDiff<0
#    flag = all(p1<=p2) and any(p1<p2) #not efficient for numba    
    return flag
	
def fib(n):
	"""Print the Fibonacci series up to n."""
	a, b = 0, 1
	while b < n:
		print b,
		a, b = b, a + b