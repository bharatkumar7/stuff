'''# -*- coding: utf-8 -*-           If you are just trying to use UTF-8 characters or don't care if they are in your code, add this line''' 
import numpy as np
import time,os,talib
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick_ohlc, candlestick2_ohlc
from matplotlib import gridspec,colors
from math import pi
import funct
import mibian
import pandas as pd
from pandas import *


stock_data=1  # 1 - Streamline data, 2 - OHLC data
tickskip=1 #1 denotes every tick
colm=0
# This must be dump 123456789
for k in range(0,1):
	if k==0:fname="code07.01.16.txt"
	#if k==0:fname="1sec_close.txt"
	
	status = 0
	pruy=0
	AccountSize=1000000
	maxtick=25
	target_profit=1.02  #Risk to Reward ratio 3:1
	stop_loss=0.97 #stop loss when entering a position
	tpf=target_profit
	tsl=stop_loss
	graph=1 #1 - Yes, 0 - None
	tick_analysis_display=1 #1 - Yes , 0 - No
	exchange_charges=8  # 16.69 for equities. 90 for option. 8 for futures
	file_format=2 # 1- OHLC format, 2 - Live ticker format(Time OHLC), 3 - Live ticker format(Range OHLC)
	tf_sec='30s'
	initcap=AccountSize
	
	
	#----------------------- File reader ------------------------------------------------------------------------------------------------------
	if file_format==1:
	 tedhi,price_open,price_high,price_low,price_close, vol=(np.loadtxt(fname, dtype=float, delimiter=',', usecols=(1,3,4,5,6,7), skiprows=1,unpack=True))
	 numb=len(price_open)
	 xoi=np.min(price_low)
	 yoi=np.max(price_high)
	 xoi=xoi-(xoi*0.05)
	 yoi=yoi+(yoi*0.05)
	 
	elif file_format==2:
	 tedhi,dummy=(np.loadtxt(fname, dtype=str, delimiter=',', usecols=(0,1),skiprows=1,unpack=True))
	 cl_price, vol=(np.loadtxt(fname, dtype=float, delimiter=',', usecols=(15,2), skiprows=1,unpack=True))
	 d={'Datetime': Series(to_datetime(tedhi.astype(int)*1e9)),
	       'price': Series((cl_price)),
	       'volume':Series((vol))}
	 df=DataFrame(d)
	 df.set_index('Datetime', inplace=True)
	 vol_sum = (df['volume'].resample(tf_sec, how='sum',fill_method='backfill',limit=0)).dropna(how='any', axis=0)
	 price_ohlc = (df['price'].resample(tf_sec, how='ohlc',fill_method='backfill',limit=0)).dropna(how='any', axis=0)
	 numb=len(price_ohlc)
	 xoi=np.min(price_ohlc.low)
	 yoi=np.max(price_ohlc.high)
	 xoi=xoi-(xoi*0.001)
	 yoi=yoi+(yoi*0.001)
	 
	elif file_format==3:
	 cl_price, vol=(np.loadtxt(fname, dtype=float, delimiter=',', usecols=(11,16), skiprows=1,unpack=True))
	 if cl_price[0]<=25:rangeper=3.5/100  #constant range percent "0.01" represents 1%
	 elif cl_price[0]<=50:rangeper=2.50/100  
	 elif cl_price[0]<=100:rangeper=2.0/100
	 elif cl_price[0]<=250:rangeper=2.5/100  
	 else: rangeper=0.001/100 
	 price_open,price_high,price_low,price_close=funct.range_bar(cl_price,vol,rangeper)
	 numb=len(price_close)
	 xoi=np.min(price_low)
	 yoi=np.max(price_high)
	 xoi=xoi-(xoi*0.001)
	 yoi=yoi+(yoi*0.001) 
	#----------------------- END OF FILE READER ------------------------------------------------------------------------------------------------------
	
	
	tdata_ltp=np.zeros(maxtick,'f')
	tdata_vol=np.zeros(maxtick,'f')
	tdata_op=np.zeros(maxtick,'f')
	tdata_hi=np.zeros(maxtick,'f')
	tdata_lo=np.zeros(maxtick,'f')
	
	buy=np.full(numb-maxtick,-10)  #full  fills all the array with the defned number "-10" this case
	bstime=np.arange(0,numb-maxtick)  #arange puts all real numbers in order 1,2,3
	sell=np.full(numb-maxtick,-10)
	short=np.full(numb-maxtick,-10)
	cover=np.full(numb-maxtick,-10)
	alltrades=np.zeros(numb-maxtick)
	PLP=np.zeros(numb-maxtick)
	PL=np.zeros(numb-maxtick)
	AS=np.zeros(numb-maxtick)
	minute=np.arange(0,numb)
	trade=0
	fp=0.0    #final percentage
	pot=AccountSize
	broker=0.0
	AS[0]=AccountSize
	kfp=0.0
	ktrades=0
	kpot=0.0
	

	tech_one=np.zeros(numb-maxtick,'f')
	tech_two=np.zeros(numb-maxtick,'f')
	tech_three=np.zeros(numb-maxtick,'f')
	flatornot=np.zeros(numb-maxtick,'f')	
	#flatornot_ema=np.zeros(numb-maxtick,'f')	

	# ---------------------- Analysis -----------------------------------
	for i in range(0,numb-maxtick):      
	 bstime[i]=i-1+maxtick
	 
	 if file_format==1 or file_format==3:
	  for j in range(i,maxtick+i):
	   tdata_ltp[j-i]=price_close[j]
	   tdata_vol[j-i]=vol[j]
	   tdata_op[j-i]=price_open[j]
	   tdata_hi[j-i]=price_high[j]
	   tdata_lo[j-i]=price_low[j]
	   
	 #time.sleep(2)
	 if file_format==2:
	  for j in range(i,maxtick+i):
	   tdata_ltp[j-i]=price_ohlc.close[j]
	   tdata_vol[j-i]=vol_sum[j]
	   tdata_op[j-i]=price_ohlc.open[j]
	   tdata_hi[j-i]=price_ohlc.high[j]
	   tdata_lo[j-i]=price_ohlc.low[j]

	   
	 float_data = [float(x) for x in tdata_ltp]
	 tdata_ltp = np.array(float_data)
	 #float_data = [float(x) for x in tdata_ltp_nifty]
	 #tdata_ltp_nifty = np.array(float_data)
	 float_data = [float(x) for x in tdata_vol]
	 tdata_vol = np.array(float_data)
	 float_data = [float(x) for x in tdata_op]
	 tdata_op = np.array(float_data)
	 float_data = [float(x) for x in tdata_hi]
	 tdata_hi = np.array(float_data)
	 float_data = [float(x) for x in tdata_lo]
	 tdata_lo = np.array(float_data)
	 

	 upordown_ltp = talib.EMA(tdata_ltp,5)
	 upordown_ltpl = talib.EMA(tdata_ltp,15)
	 upordown_ltplong = talib.EMA(upordown_ltpl,10)
	 kmav = talib.KAMA(tdata_ltp,10)
	 upordown_kmav = talib.EMA(kmav,10)
	 atrv = talib.ATR(tdata_hi, tdata_lo, tdata_ltp,timeperiod=5)
	 #upordown_ltplong = talib.EMA(tdata_ltp,40)
	 '''macd, macdsignal, macdhist = talib.MACD(tdata_ltp, fastperiod=6, slowperiod=13, signalperiod=4)
	 upordown_vol = talib.EMA(tdata_vol,5)
	 rocv = talib.ROC(tdata_ltp,5)
	 tanv = talib.TAN(tdata_ltp)
	 rsiv=talib.RSI(tdata_ltp, 14)
	 tanv = talib.TAN(tdata_ltp)'''
	 #bbuv, bbmv, bblv = talib.BBANDS(tdata_ltp, timeperiod=5, nbdevup=1, nbdevdn=1, matype=0)
	 #adxv=talib.ADX(tdata_hi, tdata_lo, tdata_ltp, timeperiod=14)
	 '''cciv=talib.CCI(tdata_hi, tdata_lo, tdata_ltp, timeperiod=5)
	 ultoscv=talib.ULTOSC(tdata_hi, tdata_lo, tdata_ltp, timeperiod1=5, timeperiod2=10, timeperiod3=15)
	 willrv=talib.WILLR(tdata_hi, tdata_lo, tdata_ltp, timeperiod=5) 
	 midpointv = talib.MIDPOINT(tdata_ltp, timeperiod=5)
	 momv=talib.MOM(tdata_ltp, timeperiod=10)
	 stfastkv, stfastdv = talib.STOCHF(tdata_hi, tdata_lo, tdata_ltp, fastk_period=5, fastd_period=3, fastd_matype=0)
	 strsifastkv, strsifastdv = talib.STOCHRSI(tdata_ltp, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
	 '''
	 hhv=np.max(tdata_hi[maxtick-14:maxtick-1])
	 llv=np.min(tdata_lo[maxtick-14:maxtick-1])
	 latest=maxtick-1
	 
	 if (kmav[latest]>1.001*kmav[latest-1]>1.001*kmav[latest-2]>1.001*kmav[latest-3]) or \
	 (kmav[latest]<0.999*kmav[latest-1]<0.999*kmav[latest-2]<0.999*kmav[latest-3]):flatornot[latest]=kmav[latest]
	 else: flatornot[latest]=0

	 Uvolt=(hhv-(2.0*atrv[latest]))
	 Dvolt=(llv+(2.0*atrv[latest]))	 
	 #---- FOR PLOT -----------
	 if graph==1:
	  tech_one[i]= Uvolt #hhv-(3*atrv[latest])#upordown_ltplong[latest]
	  tech_two[i]=Dvolt  #atrv[latest-1]
	  tech_three[i]=flatornot[latest]#adxv[latest] #atrv[latest-1]
	 #print tech_three[i]
	 #d = mibian.GK([8395, 8700, 6, 0, 11], volatility=15.7)
	 #print d.callPrice


	 # ------------------------ ENTRY --------------------------------------
	 #-------------------Uptrend ------------------------------  
	 '''if (status == 0 and tdata_hi[latest]<tdata_hi[latest-1]) and \
	 (status == 0 and tdata_hi[latest]<tdata_hi[latest-2]) and \
	 (status == 0 and tdata_hi[latest]<tdata_hi[latest-3]) and \
	 (status == 0 and upordown_ltplong[latest]>upordown_ltplong[latest-1]) :'''
 	 #if (status == 0 and upordown_ltpl[latest]>upordown_ltpl[latest-1]):
 	 if (status == 0 and tdata_ltp[latest]>tdata_op[latest] and tdata_ltp[latest-1]>tdata_op[latest-1]) and \
 	 (status == 0 and tdata_ltp[latest]>Dvolt):
	  
	  trend=1
	  if flatornot[latest]>0: trend=1
	  if(trend>=1):
	   status=1
	   lot_size,StopPrice=funct.PositionSizing_Method(status,AccountSize,stop_loss,tdata_ltp[latest],atrv[latest],Uvolt,Dvolt,alltrades,PL,i)
	   leftamt=AccountSize-(tdata_ltp[latest]*lot_size)
	   if (tick_analysis_display==1):print "  BUY - %.2f, Investment - %.2f, Lots - %.2f, Accountsize - %.2f, StopPrice - %.2f"%(tdata_ltp[latest],(lot_size*tdata_ltp[latest]),lot_size,AccountSize,StopPrice)
	   bprice=tdata_ltp[latest]
	   if graph==1:buy[i]=bprice
	   tprice=bprice


	 #-------------------Down trend (sell buy strategy)------------------------------  
	 '''if (status == 0 and tdata_hi[latest]>tdata_hi[latest-1]) and \
	 (status == 0 and tdata_hi[latest]>tdata_hi[latest-2]) and \
	 (status == 0 and tdata_hi[latest]>tdata_hi[latest-3]) and \
	 (status == 0 and upordown_ltplong[latest]<upordown_ltplong[latest-1]) :'''
 	 #if (status == 0 and upordown_ltpl[latest]<upordown_ltpl[latest-1]):
	 if (status == 0 and tdata_ltp[latest]<tdata_op[latest] and tdata_ltp[latest-1]<tdata_op[latest-1]) and \
	  (status == 0 and tdata_ltp[latest]<Uvolt):
	  trend=1
	  if flatornot[latest]>0: trend=1
	  if(trend>=1):
	   status=2
	   lot_size,StopPrice=funct.PositionSizing_Method(status,AccountSize,stop_loss,tdata_ltp[latest],atrv[latest],Uvolt,Dvolt,alltrades,PL,i)
	   leftamt=AccountSize-(tdata_ltp[latest]*lot_size)
	   if (tick_analysis_display==1):print "  Short - %.2f, Investment - %.2f, Lots - %.2f, Accountsize - %.2f, StopPrice - %.2f"%(tdata_ltp[latest],(lot_size*tdata_ltp[latest]),lot_size,AccountSize,StopPrice)
	   stprice=tdata_ltp[latest]
	   if graph==1:short[i]=stprice
	   tprice=stprice
	   
	   
	   # ------------------------ EXIT STRATEGY -----------------------------------------   
	 #------------------------  Trailing Stop Loss --------------------------- 
	# SELL trailing for uptrend buy strategy
	 '''if (status == 1 and ((tdata_ltp[latest] >= bprice*target_profit))): # if only bought
	  bprice=tdata_ltp[latest]
	  target_profit=1.0025
	  stop_loss=0.9975'''
	 #if (status == 1 and tick_analysis_display==1): print "%d, LTP - %.2f,  StopPrice - %.2f, Uvolt- %.2f"%(i+maxtick,tdata_ltp[latest],StopPrice,Uvolt)
	  # --------------------------SELL STRATEGY -------------------------------------------------------- 
	 '''if (status == 1 and tdata_ltp[latest]<Uvolt) or \
	 (status == 1 and i==(numb-maxtick-1)) or \
	 (status == 1 and  StopPrice > tdata_ltp[latest]) or \
	 (status == 1 and  stop_loss*tprice > tdata_ltp[latest]) '''
	 if (status == 1 and tdata_ltp[latest]<tdata_op[latest] and tdata_ltp[latest-1]<tdata_op[latest-1]):
	 #(status == 1 and tdata_ltp[latest] >= tprice*target_profit) :	  

	  bk_amt=((tprice+tdata_ltp[latest])*lot_size)
	  broker=(bk_amt*exchange_charges/100000.0) # 16.69 for equities. 90 for option. 8 for futures
	  PL[i]=(lot_size*(tdata_ltp[latest]-tprice))-broker
	  PLP[i]=(PL[i]*100/(tprice*lot_size))
	  fp=fp+PLP[i]
	  pot=pot+PL[i]
	  trade=trade+1
	  if PLP[i]>0.0: alltrades[i]=1
	  elif PLP[i]<0.0:alltrades[i]=-1
	  else:pass
	  if graph==1:sell[i]=tdata_ltp[latest]
	  if (tick_analysis_display==1):print "  SELL - %.2f, Percentage  %.2f"%(tdata_ltp[latest],PLP[i])
	  if (tick_analysis_display==1):print "-------------------------------"
	  stop_loss=tsl
	  target_profit=tpf
	  AccountSize=leftamt+PL[i]+(tprice*lot_size)
	  AS[i]=AccountSize
	  status=0 

	#----------------------------------------------	 
	 # Cover trailing for downtrend
	 '''if (status == 2 and ((tdata_ltp[latest]*target_profit <= stprice))): # if only bought
	  stprice=tdata_ltp[latest]
	  target_profit=1.0025
	  stop_loss=0.9975'''
	 #if (status == 2 and tick_analysis_display==1): print "%d, LTP - %.2f,  StopPrice - %.2f,Dvolt- %.2f"%(i+maxtick,tdata_ltp[latest],StopPrice,Dvolt)

	# -----------------COVER strategy  ---------------------------------------
	 '''if (status == 2 and tdata_ltp[latest]>Dvolt) or  \
	 (status == 2 and i==(numb-maxtick-1)) or \
	 (status == 2 and StopPrice < tdata_ltp[latest]) or \
	 (status == 2 and tprice < stop_loss*tdata_ltp[latest]) or \ '''
	 if (status == 2 and tdata_ltp[latest]>tdata_op[latest] and tdata_ltp[latest-1]>tdata_op[latest-1]):
	 
	 #(status == 2 and tdata_ltp[latest]*target_profit <= tprice):	 	 

	  bk_amt=((tprice+tdata_ltp[latest])*lot_size)
	  broker=(bk_amt*exchange_charges/100000.0)  # 16.69 for equities. 90 for option. 8 for futures
	  PL[i]=-((lot_size*(tdata_ltp[latest]-tprice))+broker)
	  PLP[i]=(PL[i]*100/(tprice*lot_size))
	  fp=fp+PLP[i]
	  pot=pot+PL[i]
	  trade=trade+1
	  if PLP[i]>0.0: alltrades[i]=1
	  elif PLP[i]<0.0:alltrades[i]=-1
	  else:pass
	  if graph==1:cover[i]=tdata_ltp[latest]
	  if (tick_analysis_display==1):print "  Cover - %.2f, Percentage  %.2f"%(tdata_ltp[latest],PLP[i])
	  if (tick_analysis_display==1):print "-------------------------------"
	  stop_loss=tsl
	  target_profit=tpf
	  AccountSize=leftamt+PL[i]+(tprice*lot_size)
	  AS[i]=AccountSize
	  status=0
	 
	 if AS[i]==0:AS[i]=AccountSize
	 
	 #-------------------------------------------------------------------
	 #if trade>=20 and (pot-broker>=3 or pot-broker<=-15):break
	 #if fp<=0 and trade>5: break	 
	 
	#print "-----------------------------------------------------------------"
	#print "  %d. Final Per =  %.2f,Final Amt =  %.2f, Trades = %d"%(k+1,fp, pot,trade)
	#print "-----------------------------------------------------------------"
	
	kfp=kfp+fp
	kpot=kpot+pot
	ktrades=ktrades+trade	
	#print "-------------------------------"
	#print "  %d. File= %s, Final Per =  %.2f,Final Amt =  %.2f, Trades = %d"%(k+1, fname,kfp,kpot,ktrades)
	print "  %d. Final Amt =  %.2f, Trades = %d"%(k+1, kpot,ktrades)
	#---------------Expectancy ---------------------------
	funct.Expectancy(alltrades,PLP,PL,numb-maxtick)
	
#------------- PLOTS ----------------------------------------
	if graph==1:
	#---------------- CANDLE STICK PLOTS ----------------------------------------------------------------------------
	 
	 quotes=np.zeros((numb-1,5))
	 for i in range(0,numb-1):
	  if file_format==1 or file_format==3:quotes[i]=(minute[i],price_open[i],price_high[i],price_low[i],price_close[i])	  
	  if file_format==2:quotes[i]=(minute[i],price_ohlc.open[i],price_ohlc.high[i],price_ohlc.low[i],price_ohlc.close[i])	  
	  
	 #axes = plt.gca()
	 #axes.set_xlim([0,numb])
	 #axes.set_ylim([xoi,yoi])
	 fig, ax1 = plt.subplots()
	 ax2 = ax1.twinx()
	 ax1.set_xlim([0,numb])
	 ax2.set_xlim([0,numb])
	 ax1.set_ylim([xoi,yoi])
	 ax2.set_ylim([xoi,yoi])
	 candlestick_ohlc(ax1,quotes,width=0.6, colorup=u'g', colordown=u'r', alpha=1.0)
	 #ax2.plot(minute,price_ohlc.close,'gray',bstime,buy,'ko', marker=r'$\downarrow$', markersize=20,bstime,sell,'ro',bstime,short,'r*',bstime,cover,'g*')
	 ax2.plot(bstime,buy-1,'go', marker=r'$\Uparrow$', markersize=8)
	 ax2.plot(bstime,sell+1,'ro', marker=r'$\Downarrow$', markersize=8)
	 ax2.plot(bstime,short+1,'ro', marker=r'$\blacktriangledown$', markersize=8)
	 ax2.plot(bstime,cover-1,'go', marker=r'$\blacktriangle$', markersize=8)
	 ax2.plot(bstime,tech_one,'blue',bstime,tech_two,'orange')
	 plt.fill_between(bstime,tech_three,facecolor='seagreen', alpha=0.5, interpolate=True)
	 plt.grid(b=True, which='major', color='grey', linestyle='--')
	 plt.show()
	 #---------------------------------------- NORMAL LINE PLOTS ----------------------------------------------------
	 '''
	 plt.figure(1)
	 gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
	 ax1=plt.subplot(gs[0])
	 axes = plt.gca()
	 axes.set_xlim([0,numb])
	 axes.set_ylim([xoi,yoi])
	 #figure,ax1 = plt.subplots()
	 ax2 = ax1.twinx()
	 if file_format==1 or file_format==3:ax1.plot(minute,price_close,'black',bstime,buy,'go',bstime,sell,'ro',bstime,short,'bo',bstime,cover,'ro',bstime,tech_one,'seagreen',bstime,tech_two,'lightcoral')#,bstime,tech_three,'bo')
	 if file_format==2:ax1.plot(minute,price_ohlc.close,'black',bstime,buy,'go',bstime,sell,'ro',bstime,short,'bo',bstime,cover,'ro',bstime,tech_one,'seagreen',bstime,tech_two,'lightcoral')#,bstime,tech_three,'bo')
	 #ax2.plot(bstime,tech_three,'b-')
	 plt.fill_between(bstime,tech_three,facecolor='seagreen', alpha=0.5, interpolate=True)
	 plt.grid(b=True, which='major', color='grey', linestyle='-')
	 
	 plt.subplot(gs[1])
	 axes = plt.gca()
	 axes.set_xlim([0,numb])
	 plt.plot(bstime,AS,'r-')
	 plt.grid(b=True, which='major', color='grey', linestyle='-')
	 plt.show() '''
		 
