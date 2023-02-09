##############################
# D. Jason Koskinen
# Dec. 22, 2015
#
# The exercise herein is to take a txt file
# and read in multiple data sets of X and Y
# input and calculate basic statitics quantities.
#
# Do everything in normal python arrays and then
# use the numpy converter to put the data into
# numpy arrays. Why? Because numpy arrays are hard.
#
# The data set is somehwat of a classic in statistics
# and is known as Anscombe's quartet.
##############################

import io
import numpy as np
import scipy as sp
from scipy import stats as stats

infile = io.open("../data/FranksNumbers.txt")

# Making an empty array to fill with arrays. Normally
# arrays of arrays is a bad sign, but it will work out
# fine this time.

metaArray = []

for line in infile.readlines():
    splLine = line.split()
    if len(splLine) == 3:# This is when the data sets change
        metaArray.append([])
    # end if len()
    if len(splLine) == 0 or (not splLine[0].isdigit()):
        continue
    # end not if
    
    # read in from text is generally a string so make sure
    # to explicitly cast the variable as a float
    
    x = float(splLine[0])
    y = float(splLine[1])
    metaArray[-1].append([x,y])
# end for line

# Convert the array of arrays into
# a numpy array so that nice calculations
# can be made with ease.
    
a = np.asarray(metaArray)

for i in range(0,len(a)):
    slope     = stats.linregress(a[i])[0]
    intercept =  stats.linregress(a[i])[1]
    slope = 0.48
    intercept = 3.02
    # The following code 'flattens' the tuple, which then includes
    # the x-values (1st column in the file) as part of the set over
    # which to compute the variance.
    # but actually we just want to compute the variance of the y-values.
    print("Variance for dataset %i: %f (WRONG VALUE)" % (i, np.var(a[i])))

    # The following code tells numpy (via the axis=0) to calculate
    # the variance over the
    # separate data columns (x and y), where we're mostly interested in the
    # variance in y. Also, there are two ways to think of the
    # exercise as written in the lecture notes:
    # A) you are given the line and therefore the degrees of freedom
    # are equal to the number of data points, or
    # B) the variance should be calculated using the 'unbiased'
    # estimator (shown on slide 3) which corrects the
    # degrees of freedom to be N-1. By default numpy uses
    # that the change to the degrees of freedom (ddof) is zero.
    # Ergo, for an unbiased estimator we maybe, possibly, kinda, sort of,
    # should use N-1 stead of N. Also, Troels said that he stressed this
    # in his class, so all of the students from his course should
    # know this.
    
    print("Variance for dataset %i: %f (CORRECT VALUE FOR BIASED VARIANCE)" % (i, np.var(a[i], axis=0, ddof=0)[1]))
    print("Variance for dataset %i: %f (CORRECT VALUE FOR UNBIASED VARIANCE)" % (i, np.var(a[i], axis=0, ddof=1)[1]))
    print("linear regression:  y=%0.2fx + %0.2f" % (stats.linregress(a[i])[0], stats.linregress(a[i])[1]))
    
    # just get the y-values, i.e. the observed data.
    # Note that this is more easily done if the data sets
    # have the exact numbers of entries, unlike here. The
    # difference is where you put the [:,1] and whether it
    # is necessary to 'recreate' a new numpy array.
    
    observed = sp.array(a[i])[:,1]
    expected = []
    chisq_value     = 0
    chisq_valuewith = 0

    # loop over all the data points in the data set
    # to calculate the expected values of y at each
    # value of x.
    for j in range(0, len(a[i])):
        x = a[i][j][0]
        y = x*slope + intercept
        expected.append(y)
        chisq_value += (y - observed[j])*(y - observed[j])/y
        chisq_valuewith += (y - observed[j])*(y - observed[j])/(1.22*1.22)
    # end for x,y
    
    print("chi-squared By hand:    ", chisq_value)
    print("chi-squared From SciPy: ", stats.chisquare(observed,expected)[0])
    print("chi-squared (w/ \pm 1.22 uncertainty):    ", (chisq_valuewith))
    print("Reduced chi-squared:    ", (chisq_value)/(len(a[i])))
    print("Reduced chi-squared (w/ \pm 1.22 uncertainty):    ", (chisq_valuewith)/(len(a[i])))
    print("\n\n")
    
# end for i

# There is a larger questions here related to calculation
# of the chi-squared value; we can do it, but if we do not know
# actually what the data is (money, number of cows, speed of a toddler, etc.)
# can the chi-squared or the reduced chi-squared tell use
# anything meaningful?

