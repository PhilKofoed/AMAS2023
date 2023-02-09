######################################################################
# Jan. 27, 2016 D. Jason Koskinen
#
# Doing an example from section 6.8 of
# Glen Cowan's stats book regarding
# two parameter Maximum Likelihood 'fitting'
#
#
# I use iminuit install via:
# pip install iminuit
#
# Here is some nice example code using iminuit and probfit
# http://stackoverflow.com/questions/17748600/solve-an-n-dimensional-optimisation-probl-using-iminuit
# but the code below does not use probfit
######################################################################

import io
import math
import random

import iminuit
import numpy as np
import scipy as sp
from scipy import stats as stats
from scipy import optimize

from ROOT import TCanvas, TH1F, TStyle, TH2D, TPolyLine, TLegend

savePlots = False


# This is the normalized function, i.e. PDF,
# over the x range from -1 to 1. For a different range
# the normalization factor should be recalculated.

def FFunc(alpha, beta, x):
    F = (1 + alpha*x + beta*x*x)/(2 + 2*beta/3)
    return F
# end def

# Be Sure to clear the steps_taken
# because this function declares things
# as global variables which continue to exist
# in the class for each instantiation. Note that
# for minimizations that will take many steps,
# the steps_taken variable could be a computer
# memory problem.

class CowanLLH:
    steps_taken = []
    def __init__(self, data_x):
        self.data_x = data_x
        CowanLLH.steps_taken = []
    def __call__(self, alpha, beta):
        LLHC = sum( -1.*math.log(FFunc(alpha, beta, x)) for x in self.data_x)
        CowanLLH.steps_taken.append([alpha,beta])
        return LLHC
    def GetSteps(self):
        return CowanLLH.steps_taken
    def GetData(self):
        return self.data_x
# end def

# Put the 'data' in some arrays
data_x_1 = []
data_x_2 = []

h1 = TH1F("h1", "", 100, -1, 1)

# ensure that PDF implmentation is
# correct, by plotting it in a histogram.
# Ask the class to see if in someone using
# ROOT can draw it nicely w/o the need for
# a histogram. Notice that the variables are
# _true for alpha and beta. This is to
# explicitly avoid any problems with local
# versus global declaration in the above
# class defintion of CowanLLH.

alpha_true = 0.2
beta_true  = 0.5
for i in range(0, 100):
    x = float(i-50)/50+0.005
    h1.Fill(x, FFunc(alpha_true, beta_true, x))
# end for()

##############################
# Now use the PDF from the defined
# function to do MC samples using
# the hit/miss method.
##############################

samples = 2000

# Create additional empty histograms
# for visual checking that the PDF
# sampling is somewhat correct.

h2 = h1.Clone()
h2.Clear()

h3 = h1.Clone()
h3.Clear()


for i in range(0, samples):

    x = random.uniform(-1, 1)
    y = random.uniform(0, 1)
    prob = FFunc(alpha_true, beta_true, x)

    while y > prob:
        x    = random.uniform(-1, 1)
        y    = random.uniform(0, 1)
        prob = FFunc(alpha_true, beta_true, x)
    # end while()
    data_x_1.append(x)
    h2.Fill(x)

    x = random.uniform(-1, 1)
    y = random.uniform(0, 1)
    prob = FFunc(alpha_true, beta_true, x)

    while y > prob:
        x    = random.uniform(-1, 1)
        y    = random.uniform(0, 1)
        prob = FFunc(alpha_true, beta_true, x)
    # end while()
    data_x_2.append(x)
    h3.Fill(x)

# end for()

# Scale for beauty and presentation
h2.Scale(h1.Integral()/(h2.Integral()))
h3.Scale(h1.Integral()/(h3.Integral()))


##############################
#Draw
##############################

# Make the histograms look pretty
h1.SetLineColor(1)
h3.SetLineColor(2)

tLeg = TLegend( 0.7, 0.7, 0.9, 0.9)
tLeg.AddEntry( h1, "True PDF", "l")
tLeg.AddEntry( h2, "MC Sample 1", "l")
tLeg.AddEntry( h3, "MC Sample 2", "l")


# Draw the histograms

tCan0 = TCanvas()
h1.SetStats(0)
h1.GetXaxis().SetTitle("x")
h1.GetYaxis().SetTitle("semi-normalized event rate")
h1.Draw()
h1.GetYaxis().SetRangeUser(0, 2)
h2.Draw("same")
h3.Draw("same")
tLeg.Draw()
tCan0.Update()

if savePlots:
    tCan0.SaveAs("plots/Lecture3_MLE_Scatter_TrueSamples.pdf")
# end if

##############################
# Now we have the ability to
# sample from the PDF, so we move onto
# fitting the values alpha and beta
# in this case using MINUIT routines
##############################

print
llh = CowanLLH(data_x_1)
#m_1 = iminuit.Minuit(llh, error_alpha=1)
m_1 = iminuit.Minuit(llh)
m_1.migrad()

print m_1.values

print "start the second instance"

llh_2 = CowanLLH(data_x_2)

m_2 = iminuit.Minuit(llh_2)
m_2.migrad()

print m_2.values

# Voila the fitting works!

##############################
# Now scan the likelihood space
# to see what the minimizer sees
##############################

# using the second data to see
# what is going on

# TH2 X then Y. Creating a 2D histogram
# that contains the Raster scan of the
# 2D likelihood space

#hScan = TH2D("hScan", ";#alpha;#beta", 40, -0.2, 1, 40, 0, 1)
alpha_min = -0.2
alpha_max = 0.9

beta_min = 0
beta_max = 10

NBins = 10

hScan = TH2D("hScan", ";#alpha;#beta", NBins, alpha_min, alpha_max, NBins, beta_min, beta_max)

# Gotta now convert the steps in
# alhpa and beta that MIGRAD dictates
# into a format that I can use in ROOT.

steps_x = []
steps_y = []

for mov in llh.GetSteps():
    steps_x.append(mov[0])
    steps_y.append(mov[1])
# end for()
    
trackMigrad_x = np.asarray( steps_x)
trackMigrad_y = np.asarray( steps_y)

pline = TPolyLine(len(steps_x), trackMigrad_x, trackMigrad_y)
pline.SetLineWidth(2)


for a in np.linspace(alpha_min, alpha_max, NBins):
    for b in np.linspace(beta_min, beta_max, NBins):
        hScan.Fill( a, b, llh_2(a,b))
    # end for b
# end for a

tCanScan1 = TCanvas()
hScan.Draw("COLZ")
hScan.SetStats(0)
#hScan.GetZaxis().SetRangeUser(1280, 1310)
#pline.Draw("LP*")
tCanScan1.Update()


tCanScan2 = TCanvas()
hScan.Draw("COLZ")
hScan.SetStats(0)
#hScan.GetZaxis().SetRangeUser(1280, 1310)
pline.Draw("LP*")
tCanScan2.Update()

if savePlots:
    tCanScan1.SaveAs("plots/Lecture3_MLE_Scatter_LLHScape.pdf")
    tCanScan2.SaveAs("plots/Lecture3_MLE_Scatter_LLHScapeWalk.pdf")
# end if



raw_input('Press Enter to exit')
       



