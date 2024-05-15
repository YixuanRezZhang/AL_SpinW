# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import copy as copy

class SpinW:

	def __init__(self, infile):
		with open(infile) as f:
			lines = f.readlines()

		self.hw = []
		self.hkl = []
		self.intensity = []
		self.omega = []
		HKL, HW, INTENSITY, OMEGA = False, False, False, False
		for i, line in enumerate(lines):
			if (line.startswith('\n') | line.startswith('#')):
				HKL, HW, INTENSITY = False, False, False
			else:
				if HKL:
					self.hkl.append([float(h) for h in line.split(',')])
				elif HW:
					self.hw.append([float(h) for h in line.split(',')])
				elif INTENSITY:
					self.intensity.append([float(h) for h in line.split(',')])
				elif OMEGA:
					self.omega.append([float(h) for h in line.split(',')])

			if line.startswith('# hkl'):	
				HKL = True
				#self.hkl.append([])
			if line.startswith('# omega'):	
				HKL = False
				OMEGA = True
			if line.startswith('# Evect'):	
				OMEGA = False
				HW = True
				#self.hw.append([])
			if line.startswith('# swConv'):	
				HW = False
				INTENSITY = True
				#self.intensity.append([])

		# for i in range(len(self.hw)):
		# 	self.hkl[i].append(2*self.hw[i][-1] - self.hw[i][-2])

		self.hw = np.array(self.hw).flatten()
		self.hkl = np.array(self.hkl)
		self.intensity = np.array(self.intensity)
		self.omega = np.array(self.omega)

		self.originalintensity = copy.deepcopy(self.intensity)
		self.convolute(1.5)
		self.findxlabel()


	def convolute(self, dE = 0.5):
		width = dE/np.max(self.hw)*len(self.hw)
		win = signal.gaussian(int(width)*3,width/2)
		for i in range(len(self.originalintensity[0])):
			# print(i,j)
			self.intensity[:,i] = signal.convolve(self.originalintensity[:,i], win, mode='same') /\
													 sum(win)


	def plotCuts(self, axes, **kwargs):
		axes.pcolormesh(self.xx, self.hw, self.intensity,
								rasterized = True, **kwargs)

	def findxlabel(self):
		dhkl = self.hkl[:,-1] - self.hkl[:,0]
		NonZero = dhkl > 0
		if NonZero[0] == True:   
			letter = 'H'
			dhkl /= dhkl[0]
		elif NonZero[1] == True: 
			letter = 'K'
			dhkl /= dhkl[1]
		elif NonZero[2] == True: 
			letter = 'L'
			dhkl /= dhkl[2]

		self.xdir = dhkl
		print('direction:', dhkl)
		self.xx = np.dot(dhkl,self.hkl)/np.dot(dhkl, dhkl)#/np.max(np.abs(dhkl)) # X axis values
		origin = self.hkl[:,6] - dhkl*self.xx[6]
		self.origin = origin

		try:
			label = '('
			for i in range(3):
				if np.around(dhkl[i],3) == 1:
					label += letter
				elif np.around(dhkl[i],3) == -1:
					label += '-'+letter
				elif dhkl[i] != 0:
					label += stringify(dhkl[i])+letter
				# Add origin
				if np.around(origin[i],1) > 0:
					if dhkl[i] != 0:
						label += '+'+stringify(origin[i])
					else:
						label += stringify(origin[i])
				elif np.around(origin[i],1) < 0:
					label += stringify(origin[i])
				else:
					if dhkl[i] == 0:
						label += '0'
				    
				if i < 2:
					label += ','
			label += ')'
			self.Xlabel = label
		except UnboundLocalError:
			self.Xlabel = ''


def stringify(num):
	if num.is_integer():
		return str(int(np.around(num)))
	else:
		return str(np.around(num,2))


class ConstE:

	def __init__(self, infile):
		with open(infile) as f:
			lines = f.readlines()

		self.Qx = []
		self.Qy = []
		self.intensity = []
		j = -1
		QX, QY, INTENSITY = False, False, False
		for i, line in enumerate(lines):
			if (line.startswith('\n') | line.startswith('#')):
				QX, QY, INTENSITY = False, False, False
			else:
				if QX:
					self.Qx = np.array([float(h) for h in line.split(',')])
				elif QY:
					self.Qy = np.array([float(h) for h in line.split(',')])
				elif INTENSITY:
					self.intensity[j].append([float(h) for h in line.split(',')])

			if line.startswith('# [h -h 0]'):	
				QX = True
			if line.startswith('# [l l -2l]'):	
				QX = False
				QY = True
				#self.hw.append([])
			if line.startswith('# swConv'):	
				QY = False
				INTENSITY = True
				self.intensity.append([])
				j +=1


		for i in range(len(self.intensity)):
			self.intensity[i] = np.array(self.intensity[i])

		self.originalintensity = copy.deepcopy(self.intensity)

	def plotCuts(self, axes, index, **kwargs):

		axes.pcolormesh(self.Qx, self.Qy, self.intensity[index][:,:].T, rasterized = True, **kwargs)
