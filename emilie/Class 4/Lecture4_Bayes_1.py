##############################
# D. Jason Koskinen
# Feb. 19, 2016
#
# Coding up posterior distributions
# for a Bayesian Intro
#
##############################


import io
import math
import sys

import numpy
import scipy
from scipy import special

from array import array


import ROOT
from ROOT import TCanvas, TH1F, TH1D, TLegend

savePlots = False

def prior(N):
    pr = 2
    return pr
# end def

def prior_2(N):
    pr = 1./N
    return pr
# end def

def prior_3( N, mean, sigma):
    pr = math.exp(-1./2*((N-mean)/sigma)**2)/(sigma*math.sqrt(2*math.pi))
    return pr
# end def


K   = 100
n   = 60
k_0 = 10
k_1 = 15


h0   = TH1D("posterior_0", "Posterior for N", 2400, 0, 2400)
h1   = TH1D("posterior_1", "Posterior for N", 2400, 0, 2400)
hLH0 = TH1D("LH_0", "likelihood", 2400, 0, 2400)
hLH1 = TH1D("LH_1", "likelihood", 2400, 0, 2400)
hPrior = TH1D("prior", "prior", 2400, 0, 2400)

for N in range( 100, 2400):
    #thisprior = prior_3(N, 500, 30.5*5)
    thisprior = prior_2(N)
    #thisprior = 2
    hPrior.Fill( N, thisprior)

    likelihood = scipy.special.binom(K,k_0)*scipy.special.binom(N-K,n-k_0)/scipy.special.binom(N,n)
    hLH0.Fill( N, likelihood)
    posterior  =  likelihood * thisprior
    h0.Fill( N, posterior)

    likelihood = scipy.special.binom(K,k_1)*scipy.special.binom(N-K,n-k_1)/scipy.special.binom(N,n)
    hLH1.Fill( N, likelihood)
    posterior  = likelihood * thisprior
    h1.Fill( N, posterior)
## end for N

h1.SetLineColor(2)
h1.SetLineWidth(3)

hLH1.SetLineColor(2)
hLH1.SetLineStyle(2)
hLH1.SetLineWidth(3)

hLH0.SetLineStyle(2)
hLH0.SetLineWidth(3)

hPrior.SetLineColor(1)
hPrior.SetLineWidth(2)

# Because I don't care about the likelihood normalization
h0.Scale(1./h0.Integral())
h1.Scale(1./h1.Integral())
hLH0.Scale(1./hLH0.Integral())
hLH1.Scale(1./hLH1.Integral())
hPrior.Scale(1./hPrior.Integral())

tc0 = TCanvas()
h0.SetStats(0)
h0.Draw()
h0.SetLineWidth(3)
h0.GetXaxis().SetTitle("N estimate")
h0.GetXaxis().SetRangeUser( 10, 2400)
h0.GetYaxis().SetTitle("Probability")
tc0.Update()

if savePlots:
    tc0.SaveAs("plots/Lecture4_Bayes_FishPosterior.pdf")
# end if

tleg = TLegend( 0.7, 0.65, 0.9, 0.9)
tleg.AddEntry( h0, "Posterior k=%i" % k_0, "l")
tleg.AddEntry( h1, "Posterior k=%i" % k_1, "l")
tleg.AddEntry( hLH0, "Likelihood k=%i" % k_0, "l")
tleg.AddEntry( hLH1, "Likelihood k=%i" % k_1, "l")
tleg.AddEntry( hPrior, "Prior", "l")

tleg2 = TLegend( 0.7, 0.65, 0.9, 0.9)
tleg2.AddEntry( h0, "Posterior k=%i" % k_0, "l")
tleg2.AddEntry( hLH0, "Likelihood k=%i" % k_0, "l")
tleg2.AddEntry( hPrior, "Prior", "l")


tc1 = TCanvas()
h0.SetStats(0)
h0.Draw()
h0.SetTitle("")
h0.SetLineWidth(3)
h0.GetXaxis().SetTitle("N estimate")
h0.GetXaxis().SetRangeUser( 100, 2400)
h0.GetYaxis().SetTitle("Probability")

h1.Draw("same")
hPrior.Draw("same")
hLH0.Draw("same")
hLH1.Draw("same")
tleg.Draw()
tc1.Update()

if savePlots:
    tc1.SaveAs("plots/Lecture4_Bayes_FishPosterior_2.pdf")
# end if

tc3 = TCanvas()
h0.SetStats(0)
h0.Draw()
h0.SetTitle("")
h0.SetLineWidth(3)
h0.GetXaxis().SetTitle("N estimate")
h0.GetXaxis().SetRangeUser( 100, 2400)
h0.GetYaxis().SetTitle("Probability")

hPrior.Draw("same")
hLH0.Draw("same")
tleg2.Draw()
tc3.Update()

if savePlots:
    tc3.SaveAs("plots/Lecture4_Bayes_FishPosterior_3.pdf")
# end if

print "k=%i bayesian best estimator value of N: " % k_0, h0.GetXaxis().GetBinCenter(h0.GetMaximumBin())
print "k=%i bayesian best estimator value of N: " % k_1, h1.GetXaxis().GetBinCenter(h1.GetMaximumBin())

print "k=%i likelihood best estimator value of N: " % k_0, hLH0.GetXaxis().GetBinCenter(hLH0.GetMaximumBin())
print "k=%i likelihood best estimator value of N: " % k_1, hLH1.GetXaxis().GetBinCenter(hLH1.GetMaximumBin())

print h1.GetXaxis().GetBinCenter(h1.GetMaximumBin())
print hLH1.Integral(209, 2400)

raw_input('Press Enter to exit')

    

