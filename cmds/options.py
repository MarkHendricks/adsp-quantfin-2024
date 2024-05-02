import math
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import datetime
import pandas as pd
import matplotlib.pyplot as plt

def normal_cdf(x):
    return(1 + math.erf(x/np.sqrt(2)))/2

def normal_pdf(x):
    return np.exp(-x**2/2) / np.sqrt(2*np.pi)

def bs_normargs(under=None,strike=None,T=None,rf=None,vol=None):
    d1 = (np.log(under/strike) + (rf + .5 * vol**2)*T ) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return [d1,d2]

def bs_delta(under=None,strike=None,T=None,rf=None,vol=None):
    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]

    return normal_cdf(d1)         

def bs_theta(under=None,strike=None,T=None,rf=None,vol=None):
        d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol) 
        
        temp = (- under * normal_pdf(d1) * vol)/(2*np.sqrt(T)) - rf * strike * np.exp(-rf*T) * normal_cdf(d2)
        return temp

def bs_gamma(under=None,strike=None,T=None,rf=None,vol=None):
    
    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]
    return normal_pdf(d1) / (under * vol * np.sqrt(T))


def bs_vega(under=None,strike=None,T=None,rf=None,vol=None):

    d1 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)[0]
    return normal_pdf(d1) * (under * np.sqrt(T))

def bs_price(under=None,strike=None,T=None,rf=None,vol=None,option='call'):
    d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol) 
    
    if option=='put':
        return np.exp(-rf*T)*strike * normal_cdf(-d2) - under * normal_cdf(-d1)
    else:
        return under * normal_cdf(d1) - np.exp(-rf*T)*strike * normal_cdf(d2)


def bs_rho(under=None,strike=None,T=None,rf=None,vol=None):

    d1,d2 = bs_normargs(under=under,strike=strike,T=T,rf=rf,vol=vol)
    return normal_cdf(d2) * strike * T * np.exp(-rf*T)



def bs_impvol(under=None,strike=None,T=None,rf=None,option='call',opt_price=None,volGuess=.25,showflag=False):
    func = lambda ivol: (opt_price-bs_price(vol=ivol,under=under,strike=strike,T=T,rf=rf,option=option))**2
    xstar, analytics, flag, msg = fsolve(func, volGuess, full_output=True)
    
    if showflag:
        return xstar, msg
    else:
        return xstar
    
    
def to_maturity(expiration=None, current_date=None):
    return (pd.to_datetime(expiration) - pd.to_datetime(current_date)).total_seconds()/(24*60*60)/365


def filter_stale_quotes(opt_chain):
    LDATE =opt_chain.sort_values('lastTradeDate')['lastTradeDate'].iloc[-1]
    mask = list()

    for idx in opt_chain.index:
        dt = opt_chain.loc[idx,'lastTradeDate']
        if (dt - LDATE).total_seconds()/3600 > -24:
            mask.append(idx)
    
    return mask

def clean_options(calls_raw,puts_raw,volume_threshold_quantile=.5):
    idx = filter_stale_quotes(calls_raw)
    calls = calls_raw.loc[idx,:]
    idx = filter_stale_quotes(puts_raw)
    puts = puts_raw.loc[idx,:]

    calls = calls[calls['volume'] > calls['volume'].quantile(volume_threshold_quantile)].set_index('contractSymbol')
    puts = puts[puts['volume'] > puts['volume'].quantile(volume_threshold_quantile)].set_index('contractSymbol')
    
    calls['lastTradeDate'] = calls['lastTradeDate'].dt.tz_localize(None)
    puts['lastTradeDate'] = puts['lastTradeDate'].dt.tz_localize(None)
    
    return calls, puts






def treeUnder(start,T,Nt,sigma=None,u=None,d=None):

    dt = T/Nt
    Ns = Nt+1
    
    if u is None:
        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        
    grid = np.empty((Ns,Nt+1))
    grid[:] = np.nan
    
    tree = pd.DataFrame(grid)
    
    for t in tree.columns:
        for s in range(0,t+1):
            tree.loc[s,t] = start * (d**s * u**(t-s))

    treeinfo = pd.Series({'u':u,'d':d,'Nt':Nt,'dt':dt}).T
            
    return tree, treeinfo




def treeAsset(funPayoff, treeUnder,treeInfo, Z=None, pstar=None, style='european'):
    treeV = pd.DataFrame(np.nan,index= list(range(int(treeInfo.Nt+1))),columns= list(range(int(treeInfo.Nt+1))))
    
    if style=='american':
        treeExer = treeV.copy()
    
    for t in reversed(treeV.columns):
        if t ==treeV.columns[-1]:
            for s in treeV.index:
                treeV.loc[s,t] = funPayoff(treeUnder.loc[s,t]) 
                if style=='american':
                    if treeV.loc[s,t]>0:
                        treeExer.loc[s,t] = True
                    else:
                        treeExer.loc[s,t] = False
                    
        else:
            probvec = [pstar[t-1],1-pstar[t-1]]

            for s in treeV.index[:-1]:        
                treeV.loc[s,t] = Z[t-1] * treeV.loc[[s,s+1],t+1] @ probvec
                
                if style=='american':
                    exerV = funPayoff(treeUnder.loc[s,t])
                    if exerV > treeV.loc[s,t]:
                        treeExer.loc[s,t] = True
                        treeV.loc[s,t] = exerV
                    else:
                        treeExer.loc[s,t] = False

    if style=='american':
        return treeV, treeExer
    else:
        return treeV
    
    
    
    
def bs_delta_to_strike(under,delta,sigma,T,isCall=True,r=0):
    
    if isCall:
        phi = 1
    else:
        phi = -1
        if delta > 0:
            delta *= -1
        
    strike = under * np.exp(-phi * norm.ppf(phi*delta) * sigma * np.sqrt(T) + .5*sigma**2*T)
    
    return strike
    

    