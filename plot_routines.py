from astroML.stats import binned_statistic_2d
from astropy.table import Table, Column, join, hstack, vstack
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# be sure to install autoML

## define redshift and mass bin boundaries
zbins = np.array([0.2,0.5,1.0,1.5,2.0,2.5])
mbins = np.array([9.0,9.5,10.0,10.5,11.0])
nzbins=np.size(zbins)-1
nmassbins=np.size(mbins)-1

## default plot settings
ptsize = 8
ptcolor = 'gray'
nticks=5

## font and tick label settings
# rc('text', usetex=False)
# rc('font',family='Arial')
# rc('font',size=14)

plt.rcParams['xtick.major.size']=4
plt.rcParams['xtick.major.width']=1
plt.rcParams['ytick.major.size']=4
plt.rcParams['ytick.major.width']=1

plt.rcParams['xtick.major.size']=5
plt.rcParams['xtick.major.width']=1
plt.rcParams['ytick.major.size']=5
plt.rcParams['ytick.major.width']=1

plt.rcParams['xtick.minor.size']=2.5
plt.rcParams['xtick.minor.width']=1
plt.rcParams['ytick.minor.size']=2.5
plt.rcParams['ytick.minor.width']=1

plt.rcParams['figure.autolayout']=False

## custom colormap definitions
nipy=cm.get_cmap('nipy_spectral',10)
nipy_vals=nipy(np.arange(10))
nipy_vals=nipy_vals[:-1,:]
nipy=matplotlib.colors.LinearSegmentedColormap.from_list('map',nipy_vals)

nipyr=cm.get_cmap('nipy_spectral_r',10)
nipy_vals=nipyr(np.arange(10))
nipy_vals=nipy_vals[1:,:]
nipy_r=matplotlib.colors.LinearSegmentedColormap.from_list('map',nipy_vals)

def add_axlabels(figure,xlabel,ylabel):

	figure.text(0.53, 0.04+0.04, xlabel, ha='center', va='center',size='larger')
	figure.text(0.08, 0.5, ylabel, ha='center', va='center', \
				rotation='vertical',size='larger')

def add_binlabels(figure):

	figure.text(0.21+0.005, 0.92, str(mbins[0])+' < log M* < '+str(mbins[1]), \
				ha='center', va='center',size=14)
	figure.text(0.41+0.003, 0.92, str(mbins[1])+' < log M* < '+str(mbins[2]), \
				ha='center', va='center',size=14)
	figure.text(0.61, 0.92, str(mbins[2])+' < log M* < '+str(mbins[3]), \
				ha='center', va='center',size=14)
	figure.text(0.81, 0.92, str(mbins[3])+' < log M* < '+str(mbins[4]), \
				ha='center', va='center',size=14)

	figure.text(0.92, 0.81+0.02, str(zbins[0])+' < z < '+str(zbins[1]), \
				ha='center', va='center', rotation=-90,size=14)
	figure.text(0.92, 0.61+0.0625, str(zbins[1])+' < z < '+str(zbins[2]), \
				ha='center', va='center', rotation=-90,size=14)
	figure.text(0.92, 0.4+0.115, str(zbins[2])+' < z < '+str(zbins[3]), \
				ha='center', va='center', rotation=-90,size=14)
	figure.text(0.92, 0.2+0.16, str(zbins[3])+' < z < '+str(zbins[4]), \
				ha='center', va='center', rotation=-90,size=14)
	figure.text(0.92, 0.20, str(zbins[4])+' < z < '+str(zbins[5]), \
				ha='center', va='center', rotation=-90,size=14)

def add_colorbar(figure, s, zlabel, zmin, zmax, loc=[0.13,0.08,0.25,0.015]):
				
	ax_cb = figure.add_axes(loc)

	cb = figure.colorbar(s, cax=ax_cb, orientation='horizontal', ticks = [zmin,zmax])
	cb.set_label(zlabel, labelpad=-12)

def add_uvj_box(axes, zbin, **kwargs):

	uv_cut = 1.3
	vj_cut = 1.6
	diag_slope = 0.88
	#diag_zp = 0.55
	
	diag_zp = np.array([0.69,0.59,0.49,0.49,0.49]) ## Williams+09
	cross1 = (uv_cut-diag_zp[zbin])/diag_slope
	cross2 = diag_slope*vj_cut+diag_zp[zbin]

	xx = np.linspace(-1,2.5,500)

# 	if size(axes)>1:
# 		ax=axes.flatten()
# 		for i in ax:
# 			i.plot(xx[xx<=cross1], xx[xx<=cross1]*0.0+uv_cut, **kwargs)
# 			i.plot(xx[(xx>=cross1) & (xx<=vj_cut)], \
# 					 diag_zp+xx[(xx>=cross1) & (xx<=vj_cut)]*diag_slope, **kwargs)
# 			i.plot(xx[xx>=vj_cut]*0+vj_cut, xx[xx>=vj_cut]-vj_cut+cross2, **kwargs)			 
# 		else:
	axes.plot(xx[xx<=cross1], xx[xx<=cross1]*0.0+uv_cut, **kwargs)
	axes.plot(xx[(xx>=cross1) & (xx<=vj_cut)], \
			 diag_zp[zbin]+xx[(xx>=cross1) & (xx<=vj_cut)]*diag_slope, **kwargs)
	axes.plot(xx[xx>=vj_cut]*0+vj_cut, xx[xx>=vj_cut]-vj_cut+cross2, **kwargs)			 
			

def add_line(axes, xdata, ydata, **kwargs):
	if np.size(axes)>1:
		ax=axes.flatten()
		for i in ax:
			i.plot(xdata,ydata, **kwargs)
	else:
		axes.plot(xdata,ydata,**kwargs)

def fit_line(xdata, ydata):
	
	x_fit = sm.add_constant(xdata)
	model = sm.RLM(ydata,x_fit, M=sm.robust.norms.TukeyBiweight())
	fitresults=model.fit()
	fitp = np.poly1d(fitresults.params[::-1])
   
	fitslopes=fitresults.params[-1]

	return fitp, fitslopes
	
def setup_figure(nzbins,nmassbins,xmin,xmax,ymin,ymax,xlabel,ylabel):
	fig, ax = plt.subplots(nzbins,nmassbins,sharex=True,sharey=True, \
						   subplot_kw=dict(xlim=(xmin,xmax),ylim=(ymin,ymax)), \
						   figsize=(10,10))

	fig.subplots_adjust(wspace=0.1,hspace=0.1)

	add_axlabels(fig, xlabel, ylabel)
	add_binlabels(fig)

	return fig, ax

def make_slice(redshift,mstar,z,m,condition=None):

	if condition == None:
		slice = np.where( (redshift > zbins[z]) & (redshift < zbins[z+1]) & \
						(mstar > mbins[m]) & (mstar < mbins[m+1]) )
	elif condition != None:
		slice = np.where( (redshift > zbins[z]) & (redshift < zbins[z+1]) & \
						(mstar > mbins[m]) & (mstar < mbins[m+1]) & \
						condition)
	
	return slice

def plot_errorbars(axes,xpt,ypt,xerr,yerr,**kwargs):
	if np.size(axes)>1:
		ax=axes.flatten()
		xerr=xerr.flatten()
		yerr=yerr.flatten()

		for i in range(np.size(ax)):
			ax[i].errorbar(xpt,ypt,yerr=yerr[i],xerr=xerr[i], **kwargs)
	else:
		axes.errorbar(xpt,ypt,yerr=yerr,xerr=xerr, **kwargs)

def add_text(axes,xpt,ypt,text,**kwargs):
	if np.size(axes)>1:
		ax=axes.flatten()
		text=text.flatten()

		for i in range(np.size(ax)):
			ax[i].text(xpt,ypt,text[i], **kwargs)
	else:
		axes.text(xpt,ypt,text, **kwargs)
		
		
def plot_xy(axes,xdata,ydata,ptcolor,ptsize):
	axes.scatter(xdata,ydata,\
				s=ptsize,c=ptcolor,alpha=0.75,edgecolors='none')					

def plot_xyz(axes,xdata,ydata,zdata,zmin,zmax,cmap,ptsize):
	frame=axes.scatter(xdata,ydata,\
				 s=ptsize,c=zdata,vmin=zmin,vmax=zmax,cmap=cmap,\
				 alpha=0.75,edgecolors='none')

	return frame
	
def plot_binned(axes,xdata,ydata,zdata,xmin,xmax,ymin,ymax,zmin,zmax,cmap,\
				nxbins,nybins):

	N,xedges,yedges = binned_statistic_2d(xdata,ydata, \
					  zdata, 'median',\
					  bins=(nxbins,nybins),range=[[xmin,xmax],[ymin,ymax]])
						
	frame = axes.imshow(N.T,origin='lower',\
					  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],\
					  aspect='auto', interpolation='none',cmap=cmap,\
					  vmin=zmin,vmax=zmax)

	return frame

def plot_grid(xdata, ydata, mstar, redshift, \
              xmin=None, xmax=None, ymin=None, ymax=None,\
              xlabel=None, ylabel=None,\
              ptcolor=ptcolor,ptsize=ptsize,\
              condition=None,\
              zdata=None, zmin=None, zmax=None, zlabel=None,cmap=None,\
              binned=False, nxbins=20, nybins=20, uvj=False,\
              fitslope=False, xpt=None,ypt=None,underplot=False,count=False,\
              hist=False, **kwargs):


	fig, ax = setup_figure(nzbins,nmassbins,xmin,xmax,ymin,ymax,xlabel,ylabel)
	
	for z in range(nzbins):
		for m in range(nmassbins):
			ax[z,m].locator_params(nbins=nticks)
			
			slice = make_slice(redshift,mstar,z,m,condition)
								
			plot_x = xdata[slice]
			plot_y = ydata[slice]

			if np.size(plot_x) == 0:
				continue
			
			if hist == True:
				weights=np.ones_like(plot_x)/len(plot_x)
				h=ax[z,m].hist(plot_x,bins=17,range=(xmin,xmax),weights=weights,histtype='step',color='k')
				
				if count == True:
					num = np.size(slice)
					add_text(ax[z,m],xpt,ypt,num,**kwargs)
					#print num
				
				continue
				
			if underplot==True:
				under_x = xdata[np.where(condition)]
				under_y = ydata[np.where(condition)]
				
				plot_xy(ax[z,m],under_x,under_y,ptcolor,ptsize)	
				
			if zdata == None:
				plot_xy(ax[z,m],plot_x,plot_y,ptcolor,ptsize)

			elif zdata != None:		
				plot_z = zdata[slice]
				
				if binned == False:
					s = plot_xyz(ax[z,m],plot_x,plot_y,plot_z,\
								 zmin,zmax,cmap,ptsize)  
				elif binned == True: 
					s = plot_binned(ax[z,m],plot_x,plot_y,plot_z,\
									xmin,xmax,ymin,ymax,\
									zmin,zmax,cmap,nxbins,nybins)

				add_colorbar(fig,s,zlabel,zmin,zmax)					
			
			if uvj == True:
				add_uvj_box(ax[z,m],z,lw=2,c='k')
				
			if fitslope == True:
				fit_func, slope = fit_line(plot_x,plot_y)
				xx = linspace(xmin,xmax,10)
				add_line(ax[z,m],xx,fit_func(xx),**kwargs)
				add_text(ax[z,m],xpt,ypt,str(round(slope,2)))
				
			if count == True:
				num = np.size(slice)
				add_text(ax[z,m],xpt,ypt,num)
	
	return ax
	
def oplot_grid(ax,xdata, ydata, mstar, redshift,\
			   ptcolor=ptcolor,ptsize=ptsize,\
			   condition=None,\
 			   zdata=None,zmin=None,zmax=None,cmap=None):
               
 	for z in range(nzbins):
		for m in range(nmassbins):
			
			slice = make_slice(redshift,mstar,z,m,condition)

			plot_x = xdata[slice]
			plot_y = ydata[slice]

			if np.size(plot_x) == 0:
				continue
			
			if zdata == None:
				plot_xy(ax[z,m],plot_x,plot_y,ptcolor,ptsize)

			elif zdata != None:		
				plot_z = zdata[slice]
				
				s = plot_xyz(ax[z,m],plot_x,plot_y,plot_z,\
							 zmin,zmax,cmap,ptsize)  
							 
				
