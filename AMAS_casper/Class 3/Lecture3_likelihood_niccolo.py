# #############################
# Code from Nicollo Maffezzoli
# Feb. 14, 2017
#
# Example code related to the exercises
# in Lecture 3 about likelihood and
# numerical minimizers for the NBI course
# Advanced Methods in Applied Statistics.
#
# Note that when Jason runs the same code, the 2D
# LLH scan gets overwritten for some reason. 
# #############################


from ROOT import *        
from array import array  
import math               
import numpy as np

gStyle.SetOptStat("eMR")
gStyle.SetOptFit(1111)

'''EX1'''
r = TRandom3() #period of about 10**6000
r.SetSeed(1)

Nexperiments = 1        # Number of "experiments" 
Npoints      = 500       # Number of points per experiment 

mu = 0.2
sigma = 0.1
pi = TMath.Pi()
func = TF1('func','(1/(2*pi)**0.5)*exp(-0.5*((x-[0])/[1])**2)',-10,10)
func.SetParameter(0,mu)
func.SetParameter(1,sigma)
func.SetNpx(10000)

histo = TH1F("histo", "histo from a gaussian", 1000, -3, 3)
#histo.FillRandom("func", Npoints);
valuesMC = []

# ------------------------------------
# Generate data:
# ------------------------------------
for i in range(Npoints):
	val = r.Gaus(mu,sigma)
	valuesMC.append(val)
	histo.Fill(val)

MLE = (np.average(valuesMC), np.std(valuesMC))
print 'MLE: ', MLE
canv = TCanvas("canv","", 80, 40, 650, 600)
histo.Draw()
canv.Update()


Nmu_steps = 50
min_mu = 0.1
max_mu = 0.3
delta_mu = (max_mu-min_mu)*1.0/Nmu_steps

Nsig_steps = 40
min_sig = 0.05
max_sig = 0.25
delta_sig = (max_sig-min_sig)*1.0 / Nsig_steps

'''RASTER SCAN'''
hist2D_raster = TH2F('hist2D_raster', 'LLH parameter space;mean;sigma', Nmu_steps, min_mu, max_mu, Nsig_steps, min_sig, max_sig)

for imu in range(Nmu_steps):
	mu_hypo = min_mu+(delta_mu/2) + imu*delta_mu 


	for isig in range(Nsig_steps):
		sig_hypo = min_sig+(delta_sig/2) +isig*delta_sig
		
		ullh = 0.0
		for val in valuesMC:
			ullh += log(TMath.Gaus(val, mu_hypo, sig_hypo, True))
			#ullh += log((1/(sig_hypo*(2*pi)**0.5))*exp(- ((val-mu_hypo)**2) / (2*sig_hypo**2) ) )
		
		#print (mu_hypo,sig_hypo, ullh)
		hist2D_raster.SetBinContent(imu+1, isig+1, ullh)

canvLLH = TCanvas("canvLLH","LLH Space", 80, 40, 650, 600)
hist2D_raster.Draw('COLZ')#SURF')#, TEXT SAME')
canvLLH.Update()


'''EX2'''

r = TRandom3() 
r.SetSeed(1)
alfa = 0.5
beta = 0.5
Norm = 3.0/7
Npoints = 2000
Nbins = 100
xMin = -1.0
xMax = 1.0
binWidth = (xMax-xMin)/Nbins


#def f2(x, par):
#	return par[0]*(par[1]*x[0] + (par[2]*x[0]**2)

def func(x, par):
	return par[0]*(1+ par[1]*x[0] + par[2]*(x[0]**2))

def funcHisto(x, par):
	return par[0]*par[1]*par[2]*(1+ par[3]*x[0] + par[4]*(x[0]**2))




mother = TF1("mother",func,-1.0 ,1.0, 3)
mother.SetParameters(Norm, alfa, beta)
mother.SetParNames("Norm", "Alfa", "Beta")

motherH = TF1("motherH",funcHisto,-1.0 ,1.0, 5)
motherH.SetParameters(Norm, Npoints, binWidth, alfa, beta)
motherH.SetParNames("Norm", "Npoints", "binWidth", "Alfa", "Beta")
motherH.SetLineColor(kBlue)

fitHisto = TF1("fitHisto",funcHisto,-1.0 ,1.0, 5)
fitHisto.FixParameter(0, Norm)
fitHisto.FixParameter(1, Npoints)
fitHisto.FixParameter(2, binWidth)
fitHisto.SetParameter(3, alfa)
fitHisto.SetParameter(4, beta)
fitHisto.SetParNames("Norm", "Npoints", "binWidth", "Alfa", "Beta")
fitHisto.SetLineColor(kRed)


histo2 = TH1F("Data", "Ex2 - Distribution", Nbins, -1.0, 1.0)
#histo.FillRandom("func", Npoints);
valuesMC2 = []


# ------------------------------------
# Generate data:
# ------------------------------------
for i in range(Npoints):
	val = mother.GetRandom()
	valuesMC2.append(val)
	histo2.Fill(val)


histo2.Fit('fitHisto', 'RL')


leg = TLegend(0.1,0.7,0.48,0.9);
leg.SetHeader('(alfa,beta) = (.5,.5)') # option "C" allows to center the header
leg.AddEntry(histo2,'MC',"f")
leg.AddEntry(motherH,'Mother distr.',"f")
leg.AddEntry(fitHisto,'Binned LLH fit - MINUIT MIGRAD',"f")


canv2 = TCanvas("canv2","", 80, 40, 650, 600)
histo2.Draw()
motherH.Draw('same')
leg.Draw('same')
canv2.Update()

raw_input( ' ... Press Enter to exit ... ' )
