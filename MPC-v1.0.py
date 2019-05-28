import os
import sys
import time
import math
import sqlite3
import numpy as np
from time import sleep
from scipy import signal
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

def InputLogging(u):
	ON_Time = 0
	OFF_Time = 0
	for i in range(len(u)-1):
		if str(u[i]) == str(1.0):ON_Time += 1
	OFF_Time = 48 - ON_Time
	print "ON Time = " + str(ON_Time * (0.5))
	print "OFF Time = " + str(OFF_Time * (0.5))
	conn = sqlite3.connect('/var/www/html/project/database/Energy-Consumption.db')
	print "Opened Database(Energy-Consumption) Successfully";
	sql = (ON_Time,OFF_Time,datetime.datetime.now())
	conn.execute('''INSERT INTO AIRWASHER VALUES(?,?,?)''',sql)
	conn.commit()
	print "Saved Commands Successfully to Database(Energy-Consumption)";
	conn.close()

def ReadMostUpdatedValue():
	cmd = str("cp /home/aryan/HVAC/Data-set-Real-Time.db /home/aryan/HVAC/MostUpdatedValue.db")
	os.system(cmd) 
	conn = sqlite3.connect('MostUpdatedValue.db')
	temp = 0
	hum = 0
	number = 0
	f = True
	while f:
		try:
			for i in range(11):		
				ID = i	
				sql=str("SELECT temperature,humidity from DATA WHERE id="+str(ID)+" ORDER BY last_valid_data DESC LIMIT 1")
				cursor = conn.execute(sql)
				for row in cursor:
					temp = row[0] + temp
					#hum = row[1] + hum
					number += 1		
			f = False	
		except:
			f = True
			print "Failed to Fetch(Measured Output)..."
			sleep(2)
			os.system(cmd) 
			conn = sqlite3.connect('MostUpdatedValue.db')
			pass
	conn.close()							
	T_avg = float(temp)/(number)		
	#H_avg = float(hum)/(number)	
	print "New Data(Measured Output) Received"
	return T_avg

def MPC(Mode):
    # Process Model Parameters
    K = 8                   # Gain
    tau = 3                 # Time Constant(Settling Time)
    ns = 48                 # Simulation Length
    t = np.linspace(0,ns,ns+1)
    delta_t = t[1] - t[0]

    # Controller Setting
    T_sampling = .01       # Sampling Time (30 minutes = 1800s)
    P = 3 * tau             # Prediction Horizon
    M = 6                   # Control Horizon
    maxmove = 1             # Maximum Delta_u
    
    # HVAC Simulation Model Parameters
    K_sim = 9               # Gain
    tau_sim = 2             # Time Constant(Settling Time)
    ns_sim = ns             # Simulation Length
    yp = np.zeros(ns+1)
    y_sim = np.zeros(ns+1)
    
    # Input Sequence
    u = np.zeros(ns+1)
    y_init = 20             # Initial Temperature
    
    # Setpoint Sequence
    sp = np.zeros(ns+1+2*P)
    sp[0:10] = 28
    sp[10:18] = 25
    sp[18:27] = 25
    sp[27:38] = 27
    sp[38:] = 22
    sp[5] = 21

    #  Create Plot
    plt.figure(Mode,figsize=(6,2))
    plt.ion()
    plt.show()
    if Mode == "Cooling": 
        K = -K
        K_sim = -K_sim
    elif Mode == "Warming": 
        K = K
        K_sim = K_sim

    # Process Model
    def process_model(y,t,u,K,tau):
        # Calculate Derivative
        dydt = (-y + K * u)/ tau
        return dydt

    # HVAC SImulation Model
    def Model_HVAC(y,t):
        # Calculate Derivative
        dydt = (-y + (K_sim * u_sim)) / tau_sim
        return dydt

    # Objective function      
    def objective(u_hat):
        # Prediction
        for k in range(1,2*P+1):
            if k == 1: y_hat0 = yp[i-P]
            if k <= P:
                if i-P+k < 0: u_hat[k] = 0
                else: u_hat[k] = u[i-P+k]
            elif k > P+M: u_hat[k] = u_hat[P+M]
            ts_hat = [delta_t_hat * (k-1),delta_t_hat * (k)]        
            y_hat = odeint(process_model,y_hat0,ts_hat,args=(u_hat[k],K,tau))
            y_hat0 = y_hat[-1]
            yp_hat[k] = y_hat[0] + y_init
            # Squared Error Calculation
            sp_hat[k] = sp[i] # send setpoint here
            delta_u_hat = np.zeros(2*P+1)        
            if k > P:
                delta_u_hat[k] = u_hat[k] - u_hat[k-1]
                se[k] = pow((sp_hat[k]-yp_hat[k]),2) + pow((delta_u_hat[k]),2)
        # Sum of Squared Error Calculation      
        obj = np.sum(se[P+1:])
        return obj

    # MPC Calculation
    for i in range(1,ns+1):
        if i == 1:
            y0 = y_init              # Initial Temp 
            u_sim = 0
        ts = [delta_t * (i-1),delta_t * i]
        t_sim = np.linspace(0,ns_sim+1,100)
        y_sim = odeint(Model_HVAC,0,t_sim) + y_init
        yp[i] = y_sim[i]  
        #######################################
        ######Most Updated Value function######    
        #yp[i] = ReadMostUpdatedValue()
        y0 = yp[i]
        print y0
        ####################################### 
        #######################################
        # Declare the variables in fuctions
        t_hat = np.linspace(i-P,i+P,2*P+1)
        delta_t_hat = t_hat[1] - t_hat[0]
        se = np.zeros(2*P+1)
        yp_hat = np.zeros(2*P+1)
        u_hat0 = np.zeros(2*P+1)
        sp_hat = np.zeros(2*P+1)
        obj = 0.0
        # Initial Guesses
        for k in range(1,2*P+1):
            if k <= P:
                if i-P+k < 0: u_hat0[k] = 0
                else: u_hat0[k] = u[i-P+k]
            elif k > P: u_hat0[k] = u[i]
        # Show Initial Objective
        print('Initial SSE Objective: ' + str(objective(u_hat0)))
        # Minimization Calculation
        start = time.time()
        solution = minimize(objective,u_hat0,method='SLSQP',options={'maxiter': 1e2})
        u_hat = solution.x  
        end = time.time()
        elapsed = end - start
        print('Final SSE Objective: ' + str(objective(u_hat)))
        print('Elapsed time: ' + str(elapsed) )
        delta = np.diff(u_hat)
        if i < ns:    
            if np.abs(delta[P]) >= maxmove:
                if delta[P] > 0: u[i+1] = round(u[i] + maxmove)
                else: u[i+1] = round(u[i] - maxmove)
            else: u[i+1] = round(u[i] + delta[P])
            if u[i+1] > 1: u[i+1] = 1
            if u[i+1] < 0: u[i+1] = 0
            u_sim = u[i+1]
            print "Input(MV) = ",u_sim,'\n'
            ##########################
            ######Relay function######
            ##########################
        # Plotting for MV and MO and SP
        plt.clf()
        plt.plot(t[0:i+1],sp[0:i+1],'r-',linewidth=2,label='Setpoint')
        plt.plot(t_hat[P:],sp_hat[P:],'r--',linewidth=2)
        plt.plot(t[0:i+1],yp[0:i+1],'k-',linewidth=2,label='Measured Sensors')
        plt.plot(t_hat[P:],yp_hat[P:],'k--',linewidth=2,label='Predicted Sensors')
        plt.step(t[0:i+1],u[0:i+1]*10,'b-',linewidth=2,label='Manipulated Variable')
        plt.step(t_hat[P:],u_hat[P:]*10,'b--',linewidth=2)
        plt.axvline(x=i)
        plt.axis([0, ns+P, 0, sp[0]+10])
        plt.xlabel('time',fontsize=16)
        plt.ylabel('y(t)',fontsize=16)
        plt.legend()
        plt.pause(T_sampling)
    return u

if __name__ == "__main__":
    while True:
        Input = MPC(Mode = "Warming")
        try:
            InputLogging(Input)
        except:   
            print "No database"
            sleep(5)
        print "Preparing for Start... "    
        sleep(10)    
    

