#!/usr/bin/python
import os
import sqlite3
import pandas as pd
from datetime import datetime
from datetime import timedelta  
import time
dataset = pd.DataFrame()
conn = sqlite3.connect('Data-set1.db')
print "Opened database successfully";

Delta_t = 5     #Time resolution in one day (Minute)
Speard = 2      #Indicate the added time to start time in Avg_Cmp thread (Minute) 
def Avg_Cmp(start,conn):
	stop = start + timedelta(minutes = Speard)
	start = '"' + str(start) + '"'
	stop = '"' + str(stop) + '"'
	#print start,stop
	temp_all = 0
	hum_all = 0
	number_all = 0
	for i in range(10):	
		temp = 0
		hum = 0
		number = 0
		try:
			ID = i
			cursor = conn.execute("SELECT count(*) from DATA WHERE id="+str(ID)+" and "+ stop + " > last_valid_data and last_valid_data > "+ start)	
			for row in cursor:
				number = row[0] + number
			if number!=0:
				cursor = conn.execute("SELECT temperature, humidity from DATA WHERE id="+str(ID)+" and "+ stop + "> last_valid_data and last_valid_data > "+ start)
				for row in cursor:
					temp = row[0] + temp
					hum = row[1] + hum
				temp_all = temp + temp_all
				hum_all = hum + hum_all
				number_all = number + number_all
		except:
			print "Enter Exception"
			pass
	if temp_all != 0:	
		return float(temp_all)/number_all
def Date_Finder():
	t = []
	tm =[]
	t1 = []
	t2 = []
	conn = sqlite3.connect('/home/wcl-lab/Desktop/lastserver/MPC-Movments.db')
	print "Opened database (MPC-Movments) successfully"
	cursor = conn.execute("SELECT MO, SP, MV, DATE from AIRWASHER")
	for row in cursor:
		hour = datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S.%f').hour	
		if hour >= 6 and hour <= 21:
			t.append(str(row[2])+row[3])
	conn.close()
	i = 0
	j = 0
	k = 0
	cnt = 0
	f1 = False
	d1 = datetime.strptime(t[0][1:len(t[0])], '%Y-%m-%d %H:%M:%S.%f').day
	d2 = datetime.strptime(t[1][1:len(t[1])], '%Y-%m-%d %H:%M:%S.%f').day
	while k != len(t)-1:	
		d1 = datetime.strptime(t[k][1:len(t[k])], '%Y-%m-%d %H:%M:%S.%f').day
		d2 = datetime.strptime(t[k+1][1:len(t[k+1])], '%Y-%m-%d %H:%M:%S.%f').day
		if d1 != d2:
			i = j
			j = j + cnt + 1		
			cnt = 0
			f1 = True
		else:
			cnt += 1
		if f1:
			t1.append(t[i:j])
			t2.append(t[i:j])
			f1 = False
		k += 1
	i = 0
	del t[:]
	del tm[:]
	#print t1
	while i!=len(t2)-1:
		j=0
		v =	len(t2[i])
		while j != len(t2[i]):
			if t2[i][j][0] == '1':
				#print t2[i][j]
				if t2[i][j-1][0] != '0':
					del t2[i][j]
					j = j - 1
			j += 1	
		j = 0
		while j != v:	
			if t2[i][j][0] == '1': 
				#print t2[i][j]
				del t2[i][j]
				j = j -1
			else: break
			j += 1
		i += 1
	tm = filter(None, t2)
	i = 0
	j = 0
	while i!=len(t1)-1:
		j=0	
		while j != len(t1[i]):
			#print t1[i][j]
			if t1[i][j][0] == '0' :
				if t1[i][j-1][0] == '0' and t1[i][j][0] == '0':
					del t1[i][j]
					j = j - 1
			j += 1		
		i += 1
	t = filter(None, t1)
	return t,tm
interval1,interval = Date_Finder()
print interval
g = True
f = True
for i in range (len(interval)):
	day_start = datetime.strptime(interval[i][0][1:len(interval[i][0])], '%Y-%m-%d %H:%M:%S.%f')
	day_stop = datetime.strptime(interval[i][-1][1:len(interval[i][-1])], '%Y-%m-%d %H:%M:%S.%f')
	if day_start.day == day_stop.day:
		for j in range(len(interval[i])):
			#print interval[i][j]
			if interval[i][j][0] == '1' and g:
				day_start = datetime.strptime(interval[i][j][1:len(interval[i][j])], '%Y-%m-%d %H:%M:%S.%f')
				g = False
			if interval[i][j][0] == '0':
				g = True
				day_stop = datetime.strptime(interval[i][j][1:len(interval[i][j])], '%Y-%m-%d %H:%M:%S.%f')
				print day_start,day_stop
				while 1:
					if f:
						A = Avg_Cmp(day_start,conn)
						day_start = day_start + timedelta(minutes = Delta_t)	
						B = Avg_Cmp(day_start,conn)
						f = False
						if A != None and  B != None:
							df = pd.DataFrame({'T0': [A], 'Delta_t': [Delta_t],'T1': [B]})
							dataset = dataset.append(df, ignore_index=True)	
							A = B
					day_start = day_start + timedelta(minutes = Delta_t)		
					B = Avg_Cmp(day_start,conn)	
					if A != None and  B != None:
						df = pd.DataFrame({'T0': [A], 'Delta_t': [Delta_t],'T1': [B]})
						dataset = dataset.append(df, ignore_index=True)	
						A = B	
					#day_start = day_start + timedelta(minutes = Delta_t)	
					if (day_stop - day_start).total_seconds() < Delta_t * 60:
						C = Avg_Cmp(day_stop,conn)			
						if C == None: C = Avg_Cmp(day_start,conn)
						df = pd.DataFrame({'T0': [B], 'Delta_t': [Delta_t],'T1': [C]})
						dataset = dataset.append(df, ignore_index=True)
						f = True
						break
				df = pd.DataFrame({'T0': [0], 'Delta_t': [0],'T1': [0]})			
				dataset = dataset.append(df, ignore_index=True)			
	df = pd.DataFrame({'T0': [day_start], 'Delta_t': [1],'T1': [1]})			
	dataset = dataset.append(df, ignore_index=True)		
	print dataset	
	dataset.to_pickle('System_Plant3') 				
dataset = pd.read_pickle('System_Plant3')
print dataset	
conn.commit()				
				 
