import numpy as np
from copy import deepcopy


class mantidOutput:
	def __init__(self, infile, latprms = None):
		'''Output from Mantid---assumes a 2D data format.
		Will extend this later to more.'''
		with open(infile, 'r') as f:
			firstline = f.readline().split()
			secondline = f.readline().split(':')[1].split('x')
		self.labels = firstline[1:]
		self.shape = [int(sl) for sl in secondline]

		data = np.genfromtxt(infile, unpack=True)
		xx = data[2]
		if len(np.unique(xx)) == 1:
			xx = data[3]
			yy = data[4]
			if len(np.unique(yy)) == 1:
				yy = data[5]
			if len(np.unique(xx)) == 1:
				xx = data[4]
				yy = data[5]
			self.XX = np.unique(xx)
			self.YY = np.unique(yy)

		else:
			yy = data[3+np.argwhere(np.array(self.shape[1:]) > 1)]
			self.XX = np.unique(xx)
			self.YY = np.unique(yy)
			try:
				self.ZZ1 = np.unique(data[3+np.argwhere(np.array(self.shape[1:]) == 1)[0]])
				self.ZZ2 = np.unique(data[3+np.argwhere(np.array(self.shape[1:]) == 1)[1]])
			except IndexError: 
				self.ZZ1 = 0
				self.ZZ2 = 0
		#datashape = (len(self.XX),len(self.YY))
		#self.II = data[0].reshape(datashape).T
		#self.dII = data[0].reshape(datashape).T
		datashape = [s for s in self.shape  if s > 1]
		self.II = data[0].reshape(datashape).T
		self.dII = data[1].reshape(datashape).T
		
		self.X = xx.reshape(datashape).T
		try:
			self.Y = yy.reshape(datashape).T
		except ValueError: pass


		dir1 = self.extractdirection(self.labels[2])
		dir2 = self.extractdirection(self.labels[3])
		try:
			dir3 = self.extractdirection(self.labels[4])
			dir4 = self.extractdirection(self.labels[5])
		except IndexError:
			dir3 = 0
			dir4 = 0
		# Define HKL
		if 'DeltaE' in self.labels[:4]:
			# it's a Q-E slice
			if self.labels[2] == 'DeltaE':
				self.hkl = np.outer(self.YY,dir2) + self.ZZ1*dir3 + self.ZZ2 * dir4
			elif self.labels[3] == 'DeltaE':
				self.hkl = np.outer(self.XX,dir1) + self.ZZ1*dir3 + self.ZZ2 * dir4

		else:
			try:
				hkl = np.outer(data[2].T,dir1) + np.outer(data[3].T,dir2) +\
					  np.outer(data[4].T,dir3) + np.outer(data[5].T,dir4)
			except IndexError:
				try:
					hkl = np.outer(data[2].T,dir1) + np.outer(data[3].T,dir2) +\
						  np.outer(data[4].T,dir3)
				except IndexError:
					hkl = np.outer(data[2].T,dir1) + np.outer(data[3].T,dir2)
			try:
				self.hkl = hkl.reshape((datashape[0],datashape[1],3))
			except IndexError: # it's a 1D cut
				self.hkl = hkl


	def extractdirection(self, string):
		'''str is string of [a*H, b*K, c*L]'''
		if string == 'DeltaE':
			return np.array([0,0,0])

		direct = string[1:-1].split(',')
		newdir = []
		for d in direct:
			if d in ['H', 'K','L']:
				newdir.append(1)
			elif d in ['-H', '-K','-L']:
				newdir.append(-1)
			else:
				try: 
					newdir.append(float(d.strip('H').strip('K').strip('L')))
				except ValueError:
					print(d)
		return np.array(newdir)



	def __add__(self,other):
		newobj = deepcopy(self)
		if isinstance(other, mantidOutput):
			newobj.II = self.II + other.II
			newobj.dII = np.sqrt(self.dII**2 + other.dII**2)
		else:
			print('Error. Not a MantidOutput object.')
		return newobj

	def __sub__(self,other):
		newobj = deepcopy(self)
		if isinstance(other, mantidOutput):
			newobj.II = self.II - other.II
			newobj.dII = np.sqrt(self.dII**2 + other.dII**2)
		else:
			print('Error. Not a MantidOutput object.')
		return newobj

	def __mul__(self,other):
		newobj = deepcopy(self)
		if isinstance(other, mantidOutput):
			newobj.II = self.II * other.II
			newobj.dII = np.sqrt((other.II*self.dII)**2 + (other.dII*self.II)**2)
		else:
			newobj.II = self.II * other
			newobj.dII = self.dII * other
		return newobj

	def __rmul__(self,other):
		newobj = deepcopy(self)
		if isinstance(other, mantidOutput):
			newobj.II = other.II * self.II
			newobj.dII = np.sqrt((other.II*self.dII)**2 + (other.dII*self.II)**2)
		else:
			newobj.II = other * self.II
			newobj.dII = other * self.dII
		return newobj
