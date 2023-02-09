##############################
# D. Jason Koskinen and Niccolo Maffezzoli
# Feb. 22, 2016
#
# Put together some examples of
# simple Markov Chain Monte Carlo
# tools and takeaways in code.
#
##############################

# PyMC has some included plotting features
# from matplotlib and also somehow
# sets the backend renderer to something inactive
# which restricts the pyroot canvases from being
# interactive and nice

import matplotlib
matplotlib.use('Agg')

import io
import math
import sys
import numpy 
import pymc
import scipy
from scipy import stats
from array import array
from ROOT import *
import emcee
import pystan
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt

savePlots = False

n = 100 # number of throws
k = 61  # number of times it comes up heads
p = float(k)/n # prob. from data of getting heads

# Now set the information for the priors
# and the prior function (beta-distribution)

a, b   = 5, 17
thetas = numpy.linspace(0, 1, 200)
#print 'x-scan: ', thetas
prior  = scipy.stats.beta.pdf(thetas, a, b)
#print 'PRIOR:', prior

# probability mass function (PMF) is the
# binomial analog of the usually referenced PDF
# which is the likelihood in this instance.

# I have coded it up three separate ways in the
# following so that hopefully at least one of them
# makes some sense

def BinomialLH(n, k, p):
    return scipy.special.binom( n, k)*p**k*(1-p)**(n-k)
# end BinomialLH

likelihood   = BinomialLH( n, k, thetas)
likelihood_2 = scipy.stats.binom.pmf(k, n, thetas)
likelihood_3 = scipy.stats.binom(n, thetas).pmf(k)

# Here I check if all 3 are the same
# by seeing if the ratios are equal to 1.
# Modulo some 'invalid divide' warnings from
# python is seems okay ;)

#print likelihood/likelihood_2
#print likelihood/likelihood_3

# Now I set the posterior
post = prior * likelihood

# then we do a numerical integration to generate the
# marginal likelihood to produce the final
# posterior distribution. It is trivial in
# a single dimension when we can scan across
# the space of the single parameter. It gets
# more complicated in higher dimensions and/or
# irregular PDFs when scanning is no longer
# an efficient option.

post /= (post.sum()/len(thetas))

grPrior = TGraph( len(thetas), thetas, prior)
grPost  = TGraph( len(thetas), thetas, post)
# the factor 'n' puts the likelihood onto the same normalization
# as those of the posterior and prior.
grLH    = TGraph( len(thetas), thetas, likelihood*n) 

grPrior.SetLineColor( kBlack)
grPrior.SetLineWidth(3)
grPost.SetLineColor(kRed)
grPost.SetLineWidth(3)
grLH.SetLineColor(kBlue)
grLH.SetLineWidth(3)

tleg0 = TLegend( 0.7, 0.7, 0.9, 0.9)
tleg0.AddEntry( grPrior, "Prior", "l")
tleg0.AddEntry( grPost, "Posterior", "l")
tleg0.AddEntry( grLH, "Likelihood", "l")

hBlank0 = TH1F("hBlank1", "", 100, 0, 1.0)
hBlank0.GetXaxis().SetTitle("#theta")
hBlank0.GetYaxis().SetTitle("Density")
hBlank0.GetYaxis().SetRangeUser( 0, 100)
hBlank0.SetStats(0)

tCan0 = TCanvas("c1", "c1", 650, 400)

hBlank0.Draw()
grPrior.Draw("L")
grPost.Draw("L")
grLH.Draw("L")
tleg0.Draw()
tCan0.Update()

if savePlots:
    tCan0.SaveAs("MCMC_0.pdf")
# end if

##############################
# Now us the Markov Chain
# Monte Carlo method from an external package, 
# and this will serve as a cross-check
# for when I do it by hand w/ the
# MH algorithm later. FYI, this code really should
# have been the last portion of the file, because
# it pertains to the last exercise. 
##############################

alpha = a
beta  = b

p  = pymc.Beta('p', alpha=alpha, beta=beta)
y  = pymc.Binomial('y', n=n, p=p, value=k, observed=True)
m  = pymc.Model([p, y])
mc = pymc.MCMC(m, )

mc.sample(iter=2000, burn=0)

p_samples = mc.trace('p')[:]

h1 = TH1D("hp", "", 200, 0, 1)

for i in range(0,len(p_samples)):
    h1.Fill(p_samples[i])
# end i
        
tc1 = TCanvas("tc1", "tc1", 1200, 600)
tc1.Divide(2)

tc1.cd(1)
h1.Draw()
h1.GetXaxis().SetTitle( "#theta")
tc1.Update()

grConverge = TGraph(len(p_samples), numpy.linspace(1,len(p_samples), len(p_samples)), p_samples)

tc1.cd(2)
grConverge.Draw("AP*")
grConverge.SetTitle("PyMC")
grConverge.GetXaxis().SetTitle("iterations")
grConverge.GetYaxis().SetTitle("#theta")
grConverge.GetYaxis().SetRangeUser( 0, 0.9)
tc1.Update()


if savePlots:
    tc1.SaveAs("PyMC1.pdf")
    tc1.SaveAs("PyMC1.jpg")
# end if

##############################
# Now us the emcee Markov Chain package
##############################
# NB: The definition of PDFs is a combination of TMath(ROOT) and scipy.

def log_prior(theta):  
    a,b = 5,17
    if (theta<0 or theta>1):
        print 'Prior: ', theta
        return -numpy.inf
    return numpy.log(TMath.BetaDist(theta,a,b))


def log_Binomial(theta):

    # if (theta<0 or theta>1):
    #     print 'Bino: ', theta
    #     return -numpy.inf

    k, n = 61,100

    #res = TMath.Binomial(n, k) * TMath.Power(theta, k) * TMath.Power( (1-theta), n-k ) #DOESN NOT WORK!!!!!
    res = stats.binom.logpmf(k,n,theta)
    if res != 0:
        return res
    else:
        return -numpy.inf
    
def log_post(theta):

    lp = log_prior(theta)
    if not numpy.isfinite(lp):
        return -numpy.inf
    
    return lp + log_Binomial(theta)


ndim, nwalkers = 1, 2

# Choose an initial set of positions for the walkers.
# This is a list containing nwalkers arrays of length ndim.
# Start the walkers around 0.4.
p0 = [numpy.random.rand(ndim)*0.1+0.4 for i in range(nwalkers)]
# To check the initial position of the walkers:
checkWalkers = True
if checkWalkers:
    for walker in p0:
        print 'walker: ', walker[0]


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post)

nburn  = 10
nsteps = 2000
# Clear and run the production chain.
print "Burning in ..."
# Run nburn steps as a burn-in
pos, prob, state = sampler.run_mcmc(p0, nburn)
# Reset the chain to remove the burn-in samples.
sampler.reset()
# Starting from the final position in the burn-in chain, sample for nsteps
# (rstate0 is the state of the internal random number generator)
print "Running MCMC ..."
pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state)
print ".. Emcee done!"


#---------------------------------------
#Plotting emcee results.
#---------------------------------------

# samples is a ndarray with (nwalkers x nsteps x ndim)
samples = sampler.chain[:, :, :]

walker0 = samples[0,:,0]
walker1 = samples[1,:,0]

#for i in walker0:
#    print i

grW0   = TGraph()
histW0 = TH1D("histW0", "", 200, 0, 1)
grW1 = TGraph()
histW1 = TH1D("histW1", "", 200, 0, 1)

for i in range(nsteps): 
    grW0.SetPoint(i, i, walker0[i])
    grW1.SetPoint(i, i, walker1[i])
    histW0.Fill(walker0[i])
    histW1.Fill(walker1[i])


grW0.SetMarkerColor(kRed)
grW1.SetMarkerColor(kGreen)

grW0.SetTitle('emcee - W0')
grW0.GetXaxis().SetRangeUser(0, nsteps)
grW0.GetYaxis().SetRangeUser(0, 1)

grW1.SetTitle('emcee - W1')
grW1.GetXaxis().SetRangeUser(0, nsteps)
grW1.GetYaxis().SetRangeUser(0, 1)


emceeCan = TCanvas("emceeCan", "emceeCan", 1200, 1200)
emceeCan.Divide(2,2)

emceeCan.cd(1)
histW0.Draw()
emceeCan.cd(2)
grW0.Draw('AP*')
emceeCan.cd(3)
histW1.Draw()
emceeCan.cd(4)
grW1.Draw('AP*')



##############################
# Simple Chain Example
##############################
x_0 = [100]
x_1 = [-27]

sigma = 1

iter = [0]

for i in range(100):
    # x = scipy.stats.norm.rvs(2, 1) generates x guaussianly distributed with mean 2 and sigma 1.
    x_0.append( scipy.stats.norm.rvs(0.5*x_0[-1], sigma))   
    x_1.append( scipy.stats.norm.rvs(0.5*x_1[-1], sigma))
    iter.append(float(i))
# end for

grMC_0 = TGraph(len(x_0), numpy.array(iter), numpy.array(x_0))
grMC_1 = TGraph(len(x_1), numpy.array(iter), numpy.array(x_1))

grMC_1.SetMarkerColor( 2)

tc2 = TCanvas("tc2", "tc2", 600, 550)
grMC_0.Draw("AP*")
grMC_0.GetXaxis().SetTitle("Markov Chain Iteration")
grMC_0.GetYaxis().SetRangeUser(-30, 110)
grMC_0.SetTitle("")

grMC_1.Draw("P*")
tc2.Update()
if savePlots:
    tc2.SaveAs("MarkovChainZoom_1.pdf")
# end if



##############################
# Do my own MCMC from scratch
##############################

# I didn't have to pass in n and k, but really
# why not? The target distribution is proportional
# to the posterior.

def target( LH, prior, n, k, theta):
    # first check that the Markov chain
    # sampled a physical value of the
    # theta which is a real probability, i.e. 0-1.
    if theta < 0 or theta > 1:
        return 0
    else:
        # in the following I assume that my likelihood
        # has a function PMF and that my prior has a PDF
        return LH.pmf( k, n, theta)*prior.pdf( theta)
    # end if
# end def

a, b  = 5, 17
n     = 100
k     = 61
sigma = 0.3

# Here I am creating a prior
# and likelihood distribution that are
# pretty generic and
LH    = scipy.stats.binom
prior = scipy.stats.beta( a, b)
#prior = scipy.stats.norm( a, b)

iterations = 2000
samples    = [0.5] # this is the start of my chain
iter       = [0] # this keeps track of my iterations

# Histograms for plotting the posterior distribution
# for the truncated proposal too. This will check
# to see what happens to the final target distribution
# if I re-sample (but not reject and re-iterate) non-physical
# state transitions.

hMyPost       = TH1D("hMyPost", "", 200, 0, 1)
hMyPostTrunc  = TH1D("hMyPostTrunc", "", 200, 0, 1)

for i in range(iterations):
    theta   = samples[-1]

    # My proposal distribution is a normalized gaussian
    # based only on the current value of theta, from which
    # I sample according to the PDF. Because I know that
    # a normalized gaussian is symmetric, I don't have to
    # include the probability of going from x to x'
    # compared to x' to x. Woohoo!
    theta_p = theta + scipy.stats.norm.rvs(0, sigma) 
    
    r = min(1, target(LH, prior, n, k, theta_p)/target(LH, prior, n, k, theta))
    u = numpy.random.uniform()
    
    if u < r:
        samples.append(theta_p)
    else:
        samples.append(theta)
    # end if
    hMyPost.Fill(samples[-1])
    iter.append(float(i)) # have to do this for ROOT TGraph
# end for

grMyHastings = TGraph(len(samples), numpy.array(iter), numpy.array(samples))

# now do the truncated MCMC

samples = [0.7]

for i in range(iterations):
    theta   = samples[-1]

    # My proposal distribution is a normalized gaussian
    # based only on the current value of theta, from which
    # I sample according to the PDF. Because I know that
    # a normalized gaussian is symmetric, I don't have to
    # include the probability of going from x to x'
    # compared to x' to x. Woohoo!
    
    theta_p = theta + scipy.stats.norm.rvs(0, sigma)
    
    while theta_p < 0 or theta_p > 1:
        theta_p = theta + scipy.stats.norm.rvs(0, sigma)
    #end while
    
    R = min(1, target(LH, prior, n, k, theta_p)/target(LH, prior, n, k, theta))
    r = numpy.random.uniform()
    
    if r < R:
        samples.append(theta_p)
    else:
        samples.append(theta)
    # end if
    hMyPostTrunc.Fill(samples[-1])
# end for

grMyHastingsTrunc = TGraph(len(samples), numpy.array(iter), numpy.array(samples))

tc4 = TCanvas("tc4", "tc4", 1200, 600)
tc4.Divide( 2, 1)

tc4.cd(2)
grMyHastings.Draw("AP*")
grMyHastings.SetTitle("Hand Coded Metrpolis-Hastings")
grMyHastings.GetXaxis().SetTitle("iterations")
grMyHastings.GetYaxis().SetTitle("#theta")
grMyHastings.GetYaxis().SetRangeUser( 0, 0.9)
tc4.Update()

tc4.cd(1)
hMyPost.Draw()
hMyPost.GetXaxis().SetTitle(" #theta")
tc4.Update()

if savePlots:
    tc4.SaveAs("MyHastings1.pdf")
    tc4.SaveAs("MyHastings1.jpg")
# end if


tc5 = TCanvas("tc5", "tc5", 1200, 600)
tc5.Divide( 2, 1)

tc5.cd(2)
grMyHastingsTrunc.Draw("AP")
grMyHastingsTrunc.SetTitle("Hand Coded Metrpolis-Hastings (trunc)")
grMyHastingsTrunc.GetXaxis().SetTitle("iterations")
grMyHastingsTrunc.GetYaxis().SetTitle("#theta")
grMyHastingsTrunc.GetYaxis().SetRangeUser( 0, 0.9)
tc5.Update()

tc5.cd(1)
hMyPostTrunc.Draw()
hMyPostTrunc.GetXaxis().SetTitle(" #theta")
tc5.Update()

if savePlots:
    tc5.SaveAs("MyHastings2.pdf")
    tc5.SaveAs("MyHastings2.jpg")
# end if
 


raw_input('Press Enter to exit')

    

