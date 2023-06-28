r"""Adds Tropycal functionality to Cartopy GeoAxes."""

import types
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects

def add_tropycal(ax):
    
    r"""
    Adds Tropycal plotting capability to a matplotlib.pyplot axes instance with a Cartopy projection.
    
    This axes instance must have already had a Cartopy projection added (e.g., ``projection=ccrs.PlateCarree()``).
    
    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Instance of a matplotlib axes with a Cartopy projection added.
    
    Returns
    -------
    ax
        The same axes instance is returned, with tropycal plotting functions from `tropycal.utils.cartopy_utils` added to it as methods.
    """
    
    ax.plot_storm = types.MethodType( plot_storm, ax )
    ax.plot_two = types.MethodType( plot_two, ax )
    
    return ax
    
def plot_storm(self,storm,*args,**kwargs):
    
    r"""
    Plot a Storm object on the axes instance.
    
    Parameters
    ----------
    storm : tropycal.tracks.Storm
        Instance of a Storm object to be plotted.
    
    Notes
    -----
    Besides the parameters listed above, this function behaves identically to matplotlib's default `plot()` function.
    
    It is not necessary to pass a "transform" keyword argument, as this is already assumed to be ccrs.PlateCarree().
    
    This function is already appended to an axes instance if ``ax = utils.add_tropycal(ax)`` is run beforehand. This allows this method to be called simply via ``ax.plot_storm(...)`` the same way one would call ``ax.plot(...)``.
    """
    
    #Filter to only tropical points if requested
    if 'stormtype' in kwargs.keys():
        stormtype = kwargs.pop('stormtype')
        storm = storm.sel(stormtype=stormtype)
    
    #Pass arguments to ax plot method
    self.plot(storm.lon,storm.lat,*args,**kwargs,transform=ccrs.PlateCarree())

def plot_two(self,two_dict,days=7,**kwargs):
    
    r"""
    Plot a NHC Tropical Weather Outlook (TWO) on the axes instance.
    
    Parameters
    ----------
    two_dict : dict
        Dictionary containing TWO areas and points. This dictionary can be retrieved from ``utils.get_two_current()`` or ``utils.get_two_archive()``.
    days : int
        Forecast range of TWO. Can be 2, 5 or 7 days. Default is 7.
    
    Other Parameters
    ----------------
    fontsize : int or float, optional
        Font size for formation probability labels. Default is 12.
    ms : int or float, optional
        Marker size label for invest areas. Default is 15.
    linewidth : int or float, optional
        Linewidth of TWO area. Default is 1.5.
    alpha : int or float, optional
        Opacity of TWO area fill. Default is 0.3.
    zorder : int, optional
        Optional display order on axes of TWO areas and labels.
    """
    
    #Retrieve kwargs
    fontsize = kwargs.pop('fontsize',12)
    ms = kwargs.pop('ms',15)
    alpha = kwargs.pop('alpha',0.3)
    linewidth = kwargs.pop('linewidth',1.5)
    zorder = kwargs.pop('zorder',None)
    
    #Format kwargs for zorder functions
    kwargs = {}
    if zorder != None:
        kwargs = {'zorder':zorder}
    
    #Consistency check
    if days not in [2,5,7]:
        raise ValueError("'days' must have a value of 2, 5 or 7")
    
    #Store TWO colors for reference
    color_base = {'Low':'yellow','Medium':'orange','High':'red'}
    
    #Store bbox properties
    bbox_prop = {'facecolor':'white','alpha':0.5,'edgecolor':'black','boxstyle':'round,pad=0.3'}

    #Plot areas
    if two_dict['areas'] != None:
        for record, geom in zip(two_dict['areas'].records(), two_dict['areas'].geometries()):

            #Read relevant data
            if 'RISK2DAY' in record.attributes.keys() or 'RISK5DAY' in record.attributes.keys() or 'RISK7DAY' in record.attributes.keys():
                if days == 2:
                    color = color_base.get(record.attributes['RISK2DAY'],'yellow')
                elif 'RISK5DAY' in record.attributes.keys():
                    color = color_base.get(record.attributes['RISK5DAY'],'yellow')
                else:
                    color = color_base.get(record.attributes['RISK7DAY'],'yellow')
            else:
                color = color_base.get(record.attributes['GENCAT'],'yellow')

            #Plot area
            self.add_feature(cfeature.ShapelyFeature([geom], ccrs.PlateCarree()),
                                facecolor=color, edgecolor=color, alpha=alpha, linewidth=linewidth, **kwargs)

            #Plot hatching
            self.add_feature(cfeature.ShapelyFeature([geom], ccrs.PlateCarree()),
                                facecolor='none', edgecolor='k', linewidth=linewidth*1.5, **kwargs)
            self.add_feature(cfeature.ShapelyFeature([geom], ccrs.PlateCarree()),
                                facecolor='none', edgecolor=color, linewidth=linewidth, **kwargs)

            #Add label if needed
            plot_coords = []
            if 'GENCAT' in record.attributes.keys() or two_dict['points'] == None:
                bounds = record.geometry.bounds
                plot_coords.append((bounds[0] + bounds[2]) * 0.5) #lon
                plot_coords.append(bounds[1]) #lat
                plot_coords.append(record.attributes['GENPROB'])
            else:
                found_areas = []
                for i_record, i_point in zip(two_dict['points'].records(), two_dict['points'].geometries()):
                    found_areas.append(i_record.attributes['AREA'])
                if 'AREA' in record.attributes.keys() and record.attributes['AREA'] not in found_areas:
                    bounds = record.geometry.bounds
                    plot_coords.append((bounds[0] + bounds[2]) * 0.5) #lon
                    plot_coords.append(bounds[1]) #lat
                    if days == 2:
                        plot_coords.append(record.attributes['PROB2DAY'])
                    elif 'PROB5DAY' in record.attributes.keys():
                        plot_coords.append(record.attributes['PROB5DAY'])
                    else:
                        plot_coords.append(record.attributes['PROB7DAY'])

            if len(plot_coords) > 0:
                #Transform coordinates for label
                x1, y1 = self.projection.transform_point(plot_coords[0], plot_coords[1], ccrs.PlateCarree())
                x2, y2 = self.transData.transform((x1, y1))
                x, y = self.transAxes.inverted().transform((x2, y2))

                # plot same point but using axes coordinates
                text = plot_coords[2]
                a = self.text(x,y-0.02,text,ha='center',va='top',transform=self.transAxes,fontweight='bold',fontsize=fontsize,clip_on=True,bbox=bbox_prop,**kwargs)
                a.set_path_effects([path_effects.Stroke(linewidth=0.5,foreground='w'),path_effects.Normal()])

    #Plot points
    if two_dict['points'] != None:
        for record, point in zip(two_dict['points'].records(), two_dict['points'].geometries()):

            lon = (list(point.coords)[0][0])
            lat = (list(point.coords)[0][1])

            #Determine if 5 or 7 day outlook exists
            prob_2day = record.attributes['PROB2DAY'].replace(" ","")
            risk_2day = record.attributes['RISK2DAY'].replace(" ","")
            if 'PROB5DAY' in record.attributes.keys():
                prob_5day = record.attributes['PROB5DAY'].replace(" ","")
                risk_5day = record.attributes['RISK5DAY'].replace(" ","")
            else:
                prob_5day = record.attributes['PROB7DAY'].replace(" ","")
                risk_5day = record.attributes['RISK7DAY'].replace(" ","")

            #Label area
            if days == 2:
                color = color_base.get(risk_2day,'yellow')
                text = prob_2day
            else:
                color = color_base.get(risk_5day,'yellow')
                text = prob_5day
            self.plot(lon,lat,'X',ms=ms,color=color,mec='k',mew=1.5*(ms/15.0),transform=ccrs.PlateCarree(),**kwargs)

            #Transform coordinates for label
            x1, y1 = self.projection.transform_point(lon, lat, ccrs.PlateCarree())
            x2, y2 = self.transData.transform((x1, y1))
            x, y = self.transAxes.inverted().transform((x2, y2))

            # plot same point but using axes coordinates
            a = self.text(x,y-0.03,text,ha='center',va='top',transform=self.transAxes,fontweight='bold',fontsize=fontsize,clip_on=True,bbox=bbox_prop,**kwargs)
            a.set_path_effects([path_effects.Stroke(linewidth=0.5,foreground='w'),path_effects.Normal()])
