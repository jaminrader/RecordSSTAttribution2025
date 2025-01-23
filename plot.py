import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy
import seaborn as sb
import numpy as np
ccrs = cartopy.crs
cfeature = cartopy.feature

def plot_map(D, lats, lons, title=' ', cmap='RdBu_r', vmin=-3, vmax=3, clabel=' ', coords=False, extend=None):
    fig = plt.figure(figsize = (10,4));
    ax = plt.axes(projection=ccrs.Robinson(central_longitude = 205));
    plt.pcolormesh(lons, lats, D, vmin=vmin, vmax=vmax, cmap=cmap, transform=ccrs.PlateCarree());
    ax.coastlines();
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='k', facecolor='.1'));
    cbar = plt.colorbar(location = 'bottom', shrink = .3, pad=.01,extend=extend);
    cbar.set_label(label=clabel, size=15,)
    cbar.ax.tick_params(labelsize=12)
    plt.title(title, size=15,);
    fig.tight_layout();
    if coords:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=.4, color='k', alpha = 1, linestyle='-')
        gl.xpadding = 1
        gl.xlabels_bottom = False
        gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
        gl.ylocator = mticker.FixedLocator([-45, 0, 45,])
        gl.xlabel_style = {'font': 'helvetica', 'size' : 11}
        gl.ylabel_style = {'font': 'helvetica', 'size' : 11}
    return fig

def plot_f_timeseries(yrs, Aplot, Fplot, PFplot, 
                      title=' ', yticks=None, ylim=None, 
                      predcolor='mediumorchid', texty = None, textx=1950, figsize=(6, 8/3),legend=True):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot()
    plt.plot(yrs, Aplot, '-', color='lightgray',label='temperature', linewidth=5)
    if Fplot is not None:
        plt.plot(yrs, Fplot, 'k-', linewidth=5, label='true forced', alpha=.8)
    plt.plot(yrs, PFplot, '-', color = predcolor, linewidth=5, alpha=.8, label='estimated forced')
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])
    plt.xlabel('year', size=15, color='.3')
    plt.ylabel('˚C', size=15, color='.3')
    plt.xticks(size=13)
    plt.yticks(yticks,size=13)
    ncol = 3; 
    if Fplot is None : ncol = 2
    if legend:
        plt.legend(loc='upper center', ncol=ncol)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_color('.3')
    ax.spines['left'].set_color('.3')
    ax.tick_params(colors='.3', width=2)
    sb.despine(fig=fig, trim=True)
    if texty is None:
        texty = np.max(Aplot)
    plt.text(textx, texty, title, size=15, color='.3')
    return fig

def plot_timeseries(yrs, ys, colors, labels,
                      title=' ', yticks=None, ylim=None, 
                      predcolor='mediumorchid', texty = None, textx=1950, figsize=(6, 8/3)):

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot()
    for y, color, label in zip(ys, colors, labels):
        plt.plot(yrs, y, '-', color=color, linewidth=5, label=label)
    if ylim is not None:
        plt.ylim(ylim[0],ylim[1])
    plt.xlabel('year', size=15, color='.3')
    plt.ylabel('˚C', size=15, color='.3')
    plt.xticks(size=13)
    plt.yticks(yticks,size=13)
    ncol = len(ys); 
    plt.legend(loc='upper center', ncol=ncol)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_color('.3')
    ax.spines['left'].set_color('.3')
    ax.tick_params(colors='.3', width=2)
    sb.despine(fig=fig, trim=True)
    if texty is None:
        texty = np.max(Aplot)
    plt.text(textx, texty, title, size=15, color='.3')
    return fig