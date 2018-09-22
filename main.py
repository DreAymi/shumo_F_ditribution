import sys
import numpy as np
import time
import threading
import xlrd
import numpy as np
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--problem',help='problrm number',type=int)
parser.add_argument('--method',help='used function: greedy or genetic',type=str)
args=parser.parse_args()

class process_data:
	"""
	get data from csv file. extract pucks, tickets, gates, and calculate puck's timetable
	a important method indexof_pucks_gates(elf,pucks,gates) is designed to connect pucks and gates, return a pucks.num x gates.num bool metrix -- Index.
	Index[i,j]==True present pucks[i] can use gates[j]
	"""
	def __init__(self, filepath):
		self.filepath=filepath
		self.Pucks,self.Tickets,self.Gates=self.readfile(self.filepath)
		self.selecttickets=self.select_Tickets(self.Tickets)
		self.selectpucks=self.select_Puck(self.Pucks)
		self.Timetable=self.get_timetable(self.selectpucks)

	def readfile(self,filepath):
		Pucksfile = open(filepath+'/original_data/Pucks.csv')
		Ticketsfile=open(filepath+'/original_data/Tickets.csv')
		Gatesfile=open(filepath+'/original_data/Gates.csv')
		sourcePucks=Pucksfile.readlines()
		sourceTickets=Ticketsfile.readlines()
		sourceGates=Gatesfile.readlines()
		Pucks=[]
		Tickets=[]
		Gates=[]
		for i in sourcePucks:
			Pucks.append(i.strip().split(','))
		for i in sourceTickets:
			Tickets.append(i.strip().split(','))
		for i in sourceGates:
			temp=i.strip().split(',')
			temp_p=[]
			for ii in range(len(temp)):
				if temp[ii].startswith('\"'):
					temp_p.append(temp[ii]+temp[ii+1])
				elif temp[ii].endswith('\"'):
					continue
				else:
					temp_p.append(temp[ii])
			Gates.append(temp_p)
		return np.array(Pucks),np.array(Tickets),np.array(Gates)

	def select_Tickets(self,data):
		datasize=len(data)
		print('before tickets length:',datasize)
		index=np.zeros(datasize).astype(bool)
		for ii in range(datasize):
			if data[ii][3].startswith('20') or data[ii][5].startswith('20'):
				index[ii]=True	
		selecttickets=data[index]
		print('selected tickets length:',len(selecttickets))
		return selecttickets

	def select_Puck(self,data):
		datasize=len(data)
		print('before pucks length:',datasize)
		index=np.zeros(datasize).astype(bool)
		for ii in range(datasize):
			if data[ii][1].startswith('20') or data[ii][6].startswith('20'):
				index[ii]=True
		selectpucks=data[index]
		print('selected pucks length:',len(selectpucks))
		return selectpucks

	def get_timetable(self,pucks):
		timetable=np.zeros(shape=(pucks.shape[0],2),dtype=int)
		mini=0
		mindata=1000000
		for ii in range(pucks.shape[0]):
			startday=int(pucks[ii,1][:2])
			endday=int(pucks[ii,6][:2])
			stdstarttime=(startday-19)*1440+int(pucks[ii,2].split(':')[0])*60+int(pucks[ii,2].split(':')[1])
			stdendtime=(endday-19)*1440+int(pucks[ii,7].split(':')[0])*60+int(pucks[ii,7].split(':')[1])+45
			'''
			startday=time.strptime(pucks[ii,1], "%d-%b-%y")
			endday=time.strptime(pucks[ii,6], "%d-%b-%y")
			starttime=time.strptime(pucks[ii,2], "%H:%M")
			endtime=time.strptime(pucks[ii,7], "%H:%M")
			stdstarttime=(startday.tm_mday-19)*24*60+starttime.tm_hour*60+starttime.tm_min
			stdendtime=(endday.tm_mday-19)*24*60+endtime.tm_hour*60+endtime.tm_min+45
			'''
			timetable[ii,0]=stdstarttime
			timetable[ii,1]=stdendtime
			if stdstarttime<mindata:
				mindata=stdstarttime
				mini=ii
		#print('min ii',mini)
		return timetable

	def get_data(self):
		return self.selectpucks,self.Gates,self.selecttickets,self.Timetable

	def get_timecost_index(self,Timetable):
		index=np.arange(0,Timetable.shape[0],1).astype(int)
		index_timetable=np.concatenate((index.reshape(-1,1),Timetable),axis=1)
		time_cost=Timetable[:,1]-Timetable[:,0]
		timeindex=np.argsort(time_cost)
		index_timetable=index_timetable[timeindex]
		return timeindex

	def indexof_pucks_gates(self,pucks,gates):
		Wide_body=set(['332','333','33E','33H','33L','773'])
		Narrow_body=set(['319','320','321','323','325','738','73A','73E','73H','73L'])
		pucksize=len(pucks)
		gatesize=len(gates)
		planetype=[]
		Wnum=0
		Nnum=0
		for ii in range(pucksize):
			if pucks[ii][5] in Wide_body:
				planetype.append('W')
				Wnum=Wnum+1
			elif pucks[ii][5] in Narrow_body:
				planetype.append('N')
				Nnum=Nnum+1
		indexs=np.zeros(shape=(pucksize,gatesize)).astype(bool)
		for ii in range(pucksize):
			for jj in range(gatesize):
				if (pucks[ii][4] in gates[jj][3]) and (pucks[ii][9] in gates[jj][4]) and (planetype[ii] in gates[jj][5]):
					indexs[ii][jj]=True
		print('Wnum###',Wnum)
		print('Nnum###',Nnum)
		np.savetxt('save_tempresult/Indexs.txt',indexs,fmt='%d')
		return indexs


class greedy_select:
	"""
	greedy_select contain distribute_mostpucks(splitID) which is the main greedy method of problem 1, splitID is a bool arg, control the select order of gates
	Roulette_gambling(self,pucks,gates,Index,timetable) is another method of greedy, but the usage of it is mainly on problem 2 and 3 for genetic alogrithm, 
	because this method can generate more related-less ditribution. so it's suitable for generate original gene group in genetic method. 
	"""
	def __init__(self,choosedpucks,gates,choosedtimetable,index):
		self.Gates=gates
		self.choosedpucks=choosedpucks
		self.choosedtimetable=choosedtimetable
		self.Indexs=index
	
	#if distribute gates in order: first single type(eg I I,D D)
	#then muti type(eg ID I,ID D,ID ID,...) let splitID True : self.distribute_mostpucks(True)
	#if not: let splitID False : self.distribute_mostpucks(False)
	def distribute_mostpucks(self,splitID):
		randomindex=np.arange(0,self.Gates.shape[0],1)
		np.random.shuffle(randomindex)
		alignmenttable=np.zeros(shape=(self.choosedpucks.shape[0]),dtype=int)
		alignmenttable-=1
		rest=[]	
		for ii in randomindex:
			if (self.Gates[ii][3].startswith('\"') or self.Gates[ii][4].startswith('\"')) and splitID:
				rest.append(ii)	
			else:
				index=np.arange(0,self.choosedpucks.shape[0],1).astype(int)
				subindex=index[self.Indexs[:,ii]]
				subtimetable=self.choosedtimetable[self.Indexs[:,ii]]
				subunion=np.concatenate((subindex.reshape(-1,1),subtimetable),axis=1)
				subtimeindex=np.argsort(subunion,axis=0)
				#sort by end time
				sortedunion=subunion[subtimeindex[:,2]]
				
				num=0
				first=True
				for jj in range(sortedunion.shape[0]):
					if alignmenttable[sortedunion[jj,0]]==-1:
						if first:
							alignmenttable[sortedunion[jj,0]]=ii
							lastjj=jj
							first=False
							num+=1
						elif sortedunion[jj,1]>=sortedunion[lastjj,2]:
							alignmenttable[sortedunion[jj,0]]=ii
							lastjj=jj
							num+=1	
		for ii in rest:
			index=np.arange(0,self.choosedpucks.shape[0],1).astype(int)
			subindex=index[self.Indexs[:,ii]]
			subtimetable=self.choosedtimetable[self.Indexs[:,ii]]
			subunion=np.concatenate((subindex.reshape(-1,1),subtimetable),axis=1)
			subtimeindex=np.argsort(subunion,axis=0)
			sortedunion=subunion[subtimeindex[:,2]]
			
			num=0
			first=True
			for jj in range(sortedunion.shape[0]):
				if alignmenttable[sortedunion[jj,0]]==-1:
					if first:
						alignmenttable[sortedunion[jj,0]]=ii
						lastjj=jj
						first=False
						num+=1
					elif sortedunion[jj,1]>=sortedunion[lastjj,2]:
						alignmenttable[sortedunion[jj,0]]=ii
						lastjj=jj
						num+=1
			#print('#########',num)	
		return alignmenttable

	def Roulette_gambling(self,pucks,gates,Index,timetable):
		alignmenttable=np.zeros(shape=(pucks.shape[0]),dtype=int)
		alignmenttable-=1
		gate_sub=[]
		rest=[]
		position=np.zeros(shape=(gates.shape[0],3),dtype=int)
		#position[,] : in colume 0 demention record if it is first;1 demention record lastjj; 2 demention record current position
		tag=True

		for ii in range(gates.shape[0]):
			if gates[ii][3].startswith('\"') or Gates[ii][4].startswith('\"'):
				rest.append(ii)
			index=np.arange(0,pucks.shape[0],1).astype(int)
			subindex=index[Index[:,ii]]
			subtimetable=timetable[Index[:,ii]]
			subunion=np.concatenate((subindex.reshape(-1,1),subtimetable),axis=1)
			subtimeindex=np.argsort(subunion,axis=0)
			sortedunion=subunion[subtimeindex[:,2]]
			gate_sub.append(sortedunion)

		while tag:
			countnum=0
			randomindex=np.arange(0,gates.shape[0],1)
			np.random.shuffle(randomindex)
			for ii in randomindex:
				if ii not in rest:
					if len(gate_sub[ii])>position[ii,2]:
						if alignmenttable[gate_sub[ii][position[ii,2],0]]==-1:
							if position[ii,0]==0:
								alignmenttable[gate_sub[ii][position[ii,2],0]]=ii
								position[ii,0]=1
								position[ii,1]=position[ii,2]
								position[ii,2]+=1
								
							elif gate_sub[ii][position[ii,2],1]>=gate_sub[ii][position[ii,1],2]:
								alignmenttable[gate_sub[ii][position[ii,2],0]]=ii
								position[ii,1]=position[ii,2]
								position[ii,2]+=1
								
							else:
								position[ii,2]+=1
						else:
							position[ii,2]+=1
							
					else:
						countnum+=1

			if countnum==gates.shape[0]-len(rest):
				tag=False

		tag=True
		while tag:
			countnum=0
			rest=np.array(rest,dtype=int)
			np.random.shuffle(rest)
			for ii in rest:
				if len(gate_sub[ii])>position[ii,2]:
					if alignmenttable[gate_sub[ii][position[ii,2],0]]==-1:
						if position[ii,0]==0:
							alignmenttable[gate_sub[ii][position[ii,2],0]]=ii
							position[ii,0]=1
							position[ii,1]=position[ii,2]
							position[ii,2]+=1
							
						elif gate_sub[ii][position[ii,2],1]>=gate_sub[ii][position[ii,1],2]:
							alignmenttable[gate_sub[ii][position[ii,2],0]]=ii
							position[ii,1]=position[ii,2]
							position[ii,2]+=1
							
						else:
							position[ii,2]+=1
					else:
						position[ii,2]+=1
							
				else:
					countnum+=1
			if countnum==rest.shape[0]:
				tag=False

		return alignmenttable

	def get_result(self):
		maxalign=0
		counter=0
		while counter<20:
			'''
			greedy distribution use self.distribute_mostpucks(True)
			Roulette_gambling distribution use self.Roulette_gambling(self.choosedpucks,self.Gates,self.Indexs,self.choosedtimetable)
			'''
			Alignmenttable=self.distribute_mostpucks(True)
			#Alignmenttable=self.Roulette_gambling(self.choosedpucks,self.Gates,self.Indexs,self.choosedtimetable)
			align_len=len([kk for kk in Alignmenttable if kk>=0])
			distribution_match=get_distribution_match(Alignmenttable,self.choosedpucks,self.Gates)
			used_gates=len(set(distribution_match[:,-1]))-1
			if align_len>maxalign:
				maxalign=align_len
				max_distribution_match=np.copy(distribution_match)
				max_used_gates=used_gates
				max_Alignmenttable=np.copy(Alignmenttable)
				counter=0
			else:
				counter+=1
			print('align_len',align_len)
			print('used_gates',used_gates)
		#print(max_Alignmenttable)
		print('maxalign',maxalign)
		print('max_used_gates',max_used_gates)
		saveresult(max_distribution_match)
		gate_shiyonglv(max_Alignmenttable,self.choosedtimetable,self.Gates)
		Wa,Na=getNWdistribution(max_Alignmenttable,self.Gates)
		print('Wide_plane and Narrow_plane tistribution num ',Wa,Na)


class evolution:
	"""
	this is the main part of genetic alogrithm, there's three method to generate original gene group, first method is generate_original_group(),this is a random method
	second is use greedy_object.distribute_mostpucks(True), this is the solution of problem 1
	third one is use greedy_object.Roulette_gambling(self.pucks,self.gates,self.Indexs,self.timetable) this is more suitable for generate original gene group
	before every iteration, a deal_conflict() should be called to deal with confilct. but after deal_conflict(), there are some -1
	so before select/exchange/variation a step to remove -1 is needed
	"""
	def __init__(self,index,pucks,gates,tickets,timetable,problemnum):
		self.group_num=200
		self.old_group_num=self.group_num
		self.tag=False
		self.Indexs=index
		self.pucks=pucks
		self.gates=gates
		self.tickets=tickets
		self.timetable=timetable
		self.problem=problemnum
		self.greedy_object=greedy_select(self.pucks,self.gates,self.timetable,self.Indexs)
		'''
		if you want to random generate orginal gene group, use generate_original_group(self.Indexs,self.group_num)
		if you want to generate orginal gene group with greedy selection or Roulette_gambling use generate_group_greedy(self.group_num)
		'''
		#self.group=self.generate_original_group(self.Indexs,self.group_num)
		self.group=self.generate_group_greedy(self.group_num)
		if self.problem==2:
			self.gate_weight=8
			self.time_weight=2
			self.gate_num_weight=1
		if self.problem==3:
			self.gate_weight=0.1
			self.time_weight=10
			self.gate_num_weight=1
		self.remain_best_num=4
		self.newgene_num=2*self.group_num//50
		self.exchange_gene_num=2
		self.gene_length=self.pucks.shape[0]
		self.iteration_step=0
		self.counter=0
		self.match_flight()
		self.get_group_score()
		self.rank_group()
		self.best_so_far=np.copy(self.group[:self.remain_best_num])
		self.best_so_far_score=np.copy(self.group_score[:self.remain_best_num])
		#print(self.Indexs)

	def generate_group_greedy(self,num):
		group=[]
		for ii in range(num):
			gene=self.greedy_object.Roulette_gambling(self.pucks,self.gates,self.Indexs,self.timetable)
			#gene=self.greedy_object.distribute_mostpucks(True)
			group.append(gene)
		group=np.array(group,dtype=int)
		return group


	def generate_original_group(self,index,num):
		original_group=[]
		date_num=np.arange(0,index.shape[1],1).astype(int)
		for ii in range(num):
			original_gene=np.zeros(shape=(index.shape[0]),dtype=int)
			for jj in range(index.shape[0]):
				gate_set=date_num[index[jj]]
				a=np.random.randint(low=0,high=len(gate_set),size=1)
				original_gene[jj]=gate_set[a]
			original_group.append(original_gene)
		original_group=np.array(original_group,dtype=int)
		return original_group


	def match_flight(self):
		ticketssize=self.tickets.shape[0]
		pucksize=self.pucks.shape[0]
		match_table=np.zeros(shape=(ticketssize,3),dtype=int)
		match_table=match_table-1
		for ii in range(ticketssize):
			match_table[ii,0]=int(self.tickets[ii,1])
			for jj in range(pucksize):
				if self.pucks[jj,1]==self.tickets[ii,3] and self.pucks[jj,3]==self.tickets[ii,2]:
					match_table[ii,1]=jj
					break;
			for jj in range(pucksize):
				if self.pucks[jj,6]==self.tickets[ii,5] and self.pucks[jj,8]==self.tickets[ii,4]:
					match_table[ii,2]=jj
					break;
		self.match_table=match_table
		

	def deal_conflict(self,gene):
		pucksize=self.pucks.shape[0]
		gatesize=self.gates.shape[0]
		conflict_position=np.zeros(shape=(pucksize),dtype=int)

		gate_set=[[] for i in range(gatesize)]
		for ii in range(pucksize):
			if gene[ii]!=-1:
				gate_set[int(gene[ii])].append(ii)
			else:
				conflict_position[ii]=1
		for ii in range(gatesize):
			containsize=len(gate_set[ii])
			#print('gate %d'%ii,gate_set[ii])
			if containsize>=2:
				lastpoint=0
				for jj in range(containsize-1):
					if self.timetable[gate_set[ii][jj+1],0]<self.timetable[gate_set[ii][lastpoint],1]:
						gene[gate_set[ii][jj+1]]=-1
						conflict_position[gate_set[ii][jj+1]]=1
					else:
						lastpoint=jj+1
		return gene,conflict_position


	def calculate_sroce_problem2(self,gene):
		matchsize=len([ii for ii in gene if ii==-1])
		gate_num=len(set(gene))-1
		totalcost=0
		total_matched_cost=0
		ticketssize=self.tickets.shape[0]
		cost_table=np.array([[15,20,35,40],[20,15,40,35],[35,40,20,30],[40,45,30,20]]).reshape((4,4)).astype(int)
		self.passengernum=0
		self.missedflghtnum=0
		self.huancheng_time=[]
		for ii in range(ticketssize):
			if self.match_table[ii,1]!=-1 and self.match_table[ii,2]!=-1:
				timecost=0
				errortimecost=0
				if gene[self.match_table[ii,1]]!=-1 and gene[self.match_table[ii,2]]!=-1:
					if self.pucks[self.match_table[ii,1],4]=='D':
						if self.gates[gene[self.match_table[ii,1]],1]=='T':
							keyi=0
						else:
							keyi=1
					else:
						if self.gates[gene[self.match_table[ii,1]],1]=='T':
							keyi=2
						else:
							keyi=3
					if self.pucks[self.match_table[ii,2],4]=='D':
						if self.gates[gene[self.match_table[ii,2]],1]=='T':
							keyj=0
						else:
							keyj=1
					else:
						if self.gates[gene[self.match_table[ii,2]],1]=='T':
							keyj=2
						else:
							keyj=3
					self.huancheng_time.append([cost_table[keyi,keyj],self.match_table[ii,0]])
					self.passengernum=self.passengernum+self.match_table[ii,0]
					if (cost_table[keyi,keyj]+self.timetable[self.match_table[ii,0],0])>(self.timetable[self.match_table[ii,1],1]-45):
						self.missedflghtnum=self.missedflghtnum+self.match_table[ii,0]

					timecost=cost_table[keyi,keyj]*self.match_table[ii,0]
					#print('#####',keyi,keyj,timecost,self.match_table[ii,0])
				else:
					errortimecost=50*self.match_table[ii,0]
				totalcost=totalcost+timecost+errortimecost
				total_matched_cost=total_matched_cost+timecost
		sroce=self.gate_weight*matchsize/gene.shape[0]+self.time_weight*total_matched_cost/(sum(self.match_table[:,0])*30)+self.gate_num_weight*gate_num/self.gates.shape[0]

		return sroce,totalcost,total_matched_cost


	def calculate_sroce_problem3(self,gene):
		matchsize=len([ii for ii in gene if ii==-1])
		totalcost=0
		hangban_time=0
		total_matched_cost=0
		ticketssize=self.tickets.shape[0]
		cost_table=np.array([[15,20,35,40],[20,15,40,35],[35,40,20,30],[40,45,30,20]]).reshape((4,4)).astype(int)
		jieyun_table=np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,2,1,0]]).reshape((4,4)).astype(int)
		walk_table=np.array([10,15,20,25,20,25,25,0,10,15,20,15,20,20,0,0,10,25,20,25,25,0,0,0,10,15,20,20,0,0,0,0,10,15,15,0,0,0,0,0,10,20,0,0,0,0,0,0,10]).reshape(7,7)
		self.passengernum=0
		self.missedflghtnum=0
		self.huancheng_time=[]
		self.gerenjingzhangdu=[]		
		for ii in range(ticketssize):
			if self.match_table[ii,1]!=-1 and self.match_table[ii,2]!=-1:
				timecost=0
				errortimecost=0
				if gene[self.match_table[ii,1]]!=-1 and gene[self.match_table[ii,2]]!=-1:
					if self.pucks[self.match_table[ii,1],4]=='D':
						if self.gates[gene[self.match_table[ii,1]],1]=='T':
							keyi=0
						else:
							keyi=1
					else:
						if self.gates[gene[self.match_table[ii,1]],1]=='T':
							keyi=2
						else:
							keyi=3
					if self.pucks[self.match_table[ii,2],4]=='D':
						if self.gates[gene[self.match_table[ii,2]],1]=='T':
							keyj=0
						else:
							keyj=1
					else:
						if self.gates[gene[self.match_table[ii,2]],1]=='T':
							keyj=2
						else:
							keyj=3

					if self.gates[gene[self.match_table[ii,1]],1]=='T':
						if self.gates[gene[self.match_table[ii,1]],2]=='North':
							aerai=0
						elif self.gates[gene[self.match_table[ii,1]],2]=='Center':
							aerai=1
						else:
							aerai=2
					else:
						if self.gates[gene[self.match_table[ii,1]],2]=='North':
							aerai=3
						elif self.gates[gene[self.match_table[ii,1]],2]=='Center':
							aerai=4
						elif self.gates[gene[self.match_table[ii,1]],2]=='South':
							aerai=5
						else:
							aerai=6
					if self.gates[gene[self.match_table[ii,2]],1]=='T':
						if self.gates[gene[self.match_table[ii,2]],2]=='North':
							aeraj=0
						elif self.gates[gene[self.match_table[ii,2]],2]=='Center':
							aeraj=1
						else:
							aeraj=2
					else:
						if self.gates[gene[self.match_table[ii,2]],2]=='North':
							aeraj=3
						elif self.gates[gene[self.match_table[ii,2]],2]=='Center':
							aeraj=4
						elif self.gates[gene[self.match_table[ii,2]],2]=='South':
							aeraj=5
						else:
							aeraj=6


					if aeraj>=aerai:
						walk_time=walk_table[aerai,aeraj]
					else:
						walk_time=walk_table[aeraj,aerai]
					timecost=(cost_table[keyi,keyj]+jieyun_table[keyi,keyj]*8+walk_time)*self.match_table[ii,0]

					self.huancheng_time.append([timecost,self.match_table[ii,0]])
					self.passengernum=self.passengernum+self.match_table[ii,0]
					if (timecost+self.timetable[self.match_table[ii,0],0])>(self.timetable[self.match_table[ii,1],1]-45):
						self.missedflghtnum=self.missedflghtnum+self.match_table[ii,0]

					danrenhuangbantime=(self.timetable[self.match_table[ii,1],1]-self.timetable[self.match_table[ii,0],0]-45)*self.match_table[ii,0]
					hangban_time=hangban_time+danrenhuangbantime
					self.gerenjingzhangdu.append([timecost/danrenhuangbantime,self.match_table[ii,0]])

				else:
					errortimecost=100*self.match_table[ii,0]
				totalcost=totalcost+timecost+errortimecost
				total_matched_cost=total_matched_cost+timecost
		score=self.gate_weight*matchsize/gene.shape[0]+self.time_weight*total_matched_cost/hangban_time

		return score,totalcost,total_matched_cost/hangban_time



	def get_group_score(self):
		group_score=[]
		group_totalcost=[]
		group_total_matched_cost=[]
		self.group_conflict_position=np.zeros(shape=(self.group_num,self.pucks.shape[0]),dtype=int)
		for ii in range(self.group_num):
			gene,conflict_position=np.array(self.deal_conflict(self.group[ii]),dtype=int)
			if self.problem==2:
				sroce,totalcost,total_matched_cost=self.calculate_sroce_problem2(gene)
			elif self.problem==3:
				sroce,totalcost,total_matched_cost=self.calculate_sroce_problem3(gene)
			group_score.append(sroce)
			group_totalcost.append(totalcost)
			group_total_matched_cost.append(total_matched_cost)
			self.group[ii]=gene
			self.group_conflict_position[ii]=conflict_position
		self.group_score=np.array(group_score)
		self.group_totalcost=np.array(group_totalcost)
		self.group_total_matched_cost=np.array(group_total_matched_cost)


	
	def rank_group(self):
		index=np.argsort(self.group_score)
		self.group=self.group[index]
		self.group_score=self.group_score[index]
		self.group_totalcost=self.group_totalcost[index]
		self.group_total_matched_cost=self.group_total_matched_cost[index]
		self.group_conflict_position=self.group_conflict_position[index]
		#print('totalcost :5',self.group_totalcost[:5])
		print('cost :5',self.group_total_matched_cost[:5])
						
	def exchange_gene(self,selective_gene):
		newgene=self.generate_group_greedy(self.newgene_num)
		#newgene=self.generate_original_group(self.Indexs,self.newgene_num)
		selective_gene=np.concatenate((selective_gene,newgene),axis=0)
		np.random.shuffle(selective_gene)
		for ii in range(0,self.group_num-self.remain_best_num,2):
			cross_point=np.random.randint(0,self.gene_length,size=(2*self.exchange_gene_num))
			cross_point=np.sort(cross_point)
			for jj in range(self.exchange_gene_num):
				random_data=np.random.uniform(low=0,high=1)
				if random_data<0.8:
					temp=np.copy(selective_gene[ii,cross_point[jj*2]:cross_point[jj*2+1]])
					selective_gene[ii,cross_point[jj*2]:cross_point[jj*2+1]]=selective_gene[ii+1,cross_point[jj*2]:cross_point[jj*2+1]]			
					selective_gene[ii+1,cross_point[jj*2]:cross_point[jj*2+1]]=np.copy(temp)	


	def gene_variation(self,selective_gene):
		index=np.arange(0,self.gates.shape[0],1).astype(int)
		for ii in range(self.group_num-self.remain_best_num):
			random_data=np.random.uniform(low=0,high=1,size=(self.gene_length))
			for jj in range(self.gene_length):
				if random_data[jj]<0.05:
					variationset=index[self.Indexs[jj]]
					gene_point=np.random.randint(low=0,high=len(variationset))
					selective_gene[ii,jj]=variationset[gene_point]

	
	def select_group(self):
		mixture=np.concatenate((self.group,self.group_score.reshape((-1,1))),axis=1)
		np.random.shuffle(mixture)
		self.group=mixture[:,:self.gene_length]
		self.group_score=mixture[:,-1].reshape((-1))
		selected_group=np.zeros(shape=(self.group_num-self.remain_best_num,self.gene_length))
		selected_group_score=np.zeros(shape=(self.group_num-self.remain_best_num))
		for ii in range(self.group_num-self.remain_best_num):
			a=np.random.randint(0,self.group_num)
			b=np.random.randint(0,self.group_num)
			if self.group_score[a]<self.group_score[b]:
				selected_group[ii]=np.copy(self.group[a])
				selected_group_score[ii]=np.copy(self.group_score[a])
			else:
				selected_group[ii]=np.copy(self.group[b])
				selected_group_score[ii]=np.copy(self.group_score[b])
		self.group=selected_group
		self.group_score=selected_group_score

	def inheritance(self):
		'''
		we need deal with -1, because after deal_conflict(),there are some -1, if we do select_group,exchange_gene,gene_variation directly
		 -1 will become more and more as iteration goes on. so before iteration, make sure there's no -1
		'''
		if self.tag:
			original_group=self.generate_original_group(self.Indexs,self.group_num+50)
		else:
			original_group=self.generate_original_group(self.Indexs,self.group_num)
		self.tag=False
		self.group=(original_group+1)*self.group_conflict_position+self.group

		self.select_group()
		self.exchange_gene(self.group)
		self.gene_variation(self.group)
		self.group=np.concatenate((self.group,self.best_so_far),axis=0)
		self.get_group_score()
		self.rank_group()
		self.best_so_far=np.copy(self.group[:self.remain_best_num])
		self.best_so_far_score=np.copy(self.group_score[:self.remain_best_num])
		self.group=np.copy(self.group[:self.group_num])
		self.group_score=np.copy(self.group_score[:self.group_num])

	def evolution_iteration(self):
		self.threhold=20
		while True:
			old_score=self.group_score[0]
			self.inheritance()
			new_score=self.group_score[0]
			self.iteration_step=self.iteration_step+1
			print('iteration_step:',self.iteration_step,'top5:',self.group_score[:5])
			if new_score<old_score:
				self.counter=0
			else:
				self.counter=self.counter+1
				if self.counter>self.threhold:
					self.tag=True
					self.threhold=self.threhold*1.5
					self.group_num=self.group_num-50
					self.counter=0
					if self.group_num<50:
						if self.problem==2:
							self.calculate_sroce_problem2(self.group[0].astype(int))
						if self.problem==3:
							self.calculate_sroce_problem3(self.group[0].astype(int))
						print('missed flght passenger num: ',self.missedflghtnum)
						print('total aligned passenger num: ',self.passengernum)
						num_t=np.zeros(shape=(10),dtype=int)
						for ii in range(len(self.huancheng_time)):
							if self.huancheng_time[ii][0]<=15:
								num_t[0]=num_t[0]+self.huancheng_time[ii][1]
							if self.huancheng_time[ii][0]<=20:
								num_t[1]=num_t[1]+self.huancheng_time[ii][1]
							if self.huancheng_time[ii][0]<=25:
								num_t[2]=num_t[2]+self.huancheng_time[ii][1]
							if self.huancheng_time[ii][0]<=30:
								num_t[3]=num_t[3]+self.huancheng_time[ii][1]
							if self.huancheng_time[ii][0]<=35:
								num_t[4]=num_t[4]+self.huancheng_time[ii][1]
							if self.huancheng_time[ii][0]<=40:
								num_t[5]=num_t[5]+self.huancheng_time[ii][1]
							if self.huancheng_time[ii][0]<=45:
								num_t[6]=num_t[6]+self.huancheng_time[ii][1]
							if self.huancheng_time[ii][0]<=50:
								num_t[7]=num_t[7]+self.huancheng_time[ii][1]
							if self.huancheng_time[ii][0]<=55:
								num_t[8]=num_t[8]+self.huancheng_time[ii][1]
							if self.huancheng_time[ii][0]<=60:
								num_t[9]=num_t[9]+self.huancheng_time[ii][1]
						print('shijian fenbu bi lv: ',num_t/self.passengernum)

						if self.problem==3:
							bbb=np.zeros(shape=(8))
							for ii in range(len(self.gerenjingzhangdu)):
								if self.gerenjingzhangdu[ii][0]<=0.01:
									bbb[0]=bbb[0]+self.gerenjingzhangdu[ii][1]
								if self.gerenjingzhangdu[ii][0]<=0.015:
									bbb[1]=bbb[1]+self.gerenjingzhangdu[ii][1]
								if self.gerenjingzhangdu[ii][0]<=0.02:
									bbb[2]=bbb[2]+self.gerenjingzhangdu[ii][1]
								if self.gerenjingzhangdu[ii][0]<=0.025:
									bbb[3]=bbb[3]+self.gerenjingzhangdu[ii][1]
								if self.gerenjingzhangdu[ii][0]<=0.03:
									bbb[4]=bbb[4]+self.gerenjingzhangdu[ii][1]
								if self.gerenjingzhangdu[ii][0]<=0.035:
									bbb[5]=bbb[5]+self.gerenjingzhangdu[ii][1]
								if self.gerenjingzhangdu[ii][0]<=0.04:
									bbb[6]=bbb[6]+self.gerenjingzhangdu[ii][1]
								if self.gerenjingzhangdu[ii][0]>0.045:
									bbb[7]=bbb[7]+self.gerenjingzhangdu[ii][1]
							print('jing zhang du ren shu fen bu:',bbb)

						return self.group[0].astype(int),self.group_total_matched_cost[0]

	def get_result(self):
		Alignmenttable,cost=self.evolution_iteration()
		alignlen=len([kk for kk in Alignmenttable if kk>=0])
		distribution_match=get_distribution_match(Alignmenttable,self.pucks,self.gates)
		used_gates=len(set(distribution_match[:,-1]))-1
		#print(Alignmenttable)
		print('align num',alignlen)
		print('used gates num',used_gates)
		if self.problem==3:
			print('zong ti jingzhang du: ',cost)
		elif self.problem==2:
			print('liucheng time cost: ',cost)
		saveresult(distribution_match)
		gate_shiyonglv(Alignmenttable,self.timetable,self.gates)
		Wa,Na=getNWdistribution(Alignmenttable,self.gates)
		print('Wide_plane and Narrow_plane tistribution num ',Wa,Na)


#tongji feiji shi jian fen bu
def distribution(timetable):
	starttime=np.min(timetable)
	endtime=np.max(timetable)
	print('##################',starttime,endtime)
	size=(endtime-starttime)//5
	statistic_time=np.zeros(shape=(size),dtype=int)
	for ii in range(size):
		num=0
		for jj in range(timetable.shape[0]):
			if timetable[jj][0]<=starttime+ii*5+2.5 and timetable[jj][1]>=starttime+ii*5+2.5:
				num+=1
		statistic_time[ii]=num
	statistic_time=statistic_time.reshape(-1,1)
	np.savetxt('save_tempresult/statistic_time.txt',statistic_time,fmt='%d')

def get_distribution_match(alignmenttable,pucks,gates):
	puck=[]
	gate=[]
	for ii in range(alignmenttable.shape[0]):
		if alignmenttable[ii]!=-1:
			puck.append(pucks[ii][0])
			gate.append(gates[alignmenttable[ii],0])
		else:
			puck.append(pucks[ii][0])
			gate.append('lawn')
	puck=np.array(puck).reshape(-1,1)
	gate=np.array(gate).reshape(-1,1)
	distribution_match=np.concatenate((puck,gate),axis=1)
	return distribution_match

def saveresult(max_distribution_match):
	puckindex=np.argsort(max_distribution_match,axis=0)
	max_distribution_match=max_distribution_match[puckindex[:,0]]
	#print(max_distribution_match)
	np.savetxt('save_tempresult/distribution_orderby_selectpuckID.txt',max_distribution_match[:,1],fmt='%s')

def getNWdistribution(alignment,gates):
	Walignednum=0
	Nalignednum=0
	for ii in range(alignment.shape[0]):
		if alignment[ii]!=-1:
			if gates[alignment[ii],5]=='W':
				Walignednum=Walignednum+1
			elif gates[alignment[ii],5]=='N':
				Nalignednum=Nalignednum+1
	return Walignednum,Nalignednum

def gate_shiyonglv(alignmenttable,timetable,gates):
	Tset=[]
	Sset=[]
	gate_time=np.zeros(shape=(gates.shape[0]))
	for ii in alignmenttable:
		if ii!=-1:
			if gates[ii,1]==('T'):
				Tset.append(ii)
			elif gates[ii,1]==('S'):
				Sset.append(ii)

	gate_set=[[] for i in range(gates.shape[0])]
	for ii in range(alignmenttable.shape[0]):
		if alignmenttable[ii]!=-1:
			gate_set[alignmenttable[ii]].append(ii)
	
	for ii in range(gates.shape[0]):
		timesum=0
		for kk in gate_set[ii]:
			endtime=timetable[kk,1]-45
			starttime=timetable[kk,0]
			if endtime>2880:
				endtime=2880
			if starttime<1440:
				starttime=1440
			timesum=timesum+(endtime-starttime)
		gate_time[ii]=timesum
	gate_time=gate_time/1440
	result=[]
	for ii in range(gates.shape[0]):
		result.append([gates[ii,0],gate_time[ii]])
	print('T Gate use ',len(set(Tset)))
	print('S Gate use ',len(set(Sset)))
	np.savetxt('save_tempresult/gate_shiyonglv.txt',result,fmt='%s')


if __name__=='__main__':
	bathpath=sys.path[0]
	problem=args.problem
	method=args.method

	process_data_object=process_data(bathpath)
	selectpucks,Gates,selecttickets,Timetable=process_data_object.get_data()
	distribution(Timetable)

	if method=='greedy':
		Timecost_index=process_data_object.get_timecost_index(Timetable)
		selectpucks=selectpucks[Timecost_index]
		choosedpucks=np.copy(selectpucks)
		Timetable=Timetable[Timecost_index]
		choosedtimetable=np.copy(Timetable)
		Indexs=process_data_object.indexof_pucks_gates(choosedpucks,Gates)
		greedy_object=greedy_select(choosedpucks,Gates,choosedtimetable,Indexs)
		greedy_object.get_result()


	elif method=='genetic':
		Timetable_index=np.argsort(Timetable[:,1])
		selectpucks=selectpucks[Timetable_index]
		choosedpucks=np.copy(selectpucks)
		Timetable=Timetable[Timetable_index]
		choosedtimetable=np.copy(Timetable)
		Indexs=process_data_object.indexof_pucks_gates(selectpucks,Gates)
		genetic_object=evolution(Indexs,choosedpucks,Gates,selecttickets,choosedtimetable,problem)	
		genetic_object.get_result()

	else:
		print('command args error')



