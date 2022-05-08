r"""A collection of relevant constants used throughout Tropycal scripts."""

#Tropical or subtropical storm types (including depressions)
TROPICAL_STORM_TYPES = frozenset(['SD','SS','TD','TS','HU'])

#Tropical or subtropical storm types (excluding depressions)
NAMED_TROPICAL_STORM_TYPES = frozenset(['SS','TS','HU'])

#Tropical only storm types
TROPICAL_ONLY_STORM_TYPES = frozenset(['TD','TS','HU'])

#Tropical only storm types
SUBTROPICAL_ONLY_STORM_TYPES = frozenset(['SD','SS'])

#Standard 00/06/12/18 UTC hours
STANDARD_HOURS = frozenset(['0000','0600','1200','1800'])

#Accepted basins
ALL_BASINS = frozenset(['north_atlantic','east_pacific','west_pacific','north_indian','south_atlantic','south_indian','australia','south_pacific'])

#Accepted NHC basins
NHC_BASINS = frozenset(['north_atlantic','east_pacific'])

#NHC Cone Radii, in nautical miles
CONE_SIZE_ATL = {
    2022: [16,26,39,52,67,84,100,142,200],
    2021: [16,27,40,55,69,86,102,148,200],
    2020: [16,26,41,55,69,86,103,151,196],
    2019: [16,26,41,54,68,102,151,198],
    2018: [16,26,43,56,74,103,151,198],
    2017: [16,29,45,63,78,107,159,211],
    2016: [16,30,49,66,84,115,165,237],
    2015: [16,32,52,71,90,122,170,225],
    2014: [16,33,52,72,92,125,170,226],
    2013: [16,33,52,72,92,128,177,229],
    2012: [16,36,56,75,95,141,180,236],
    2011: [16,36,59,79,98,144,190,239],
    2010: [16,36,62,85,108,161,220,285],
    2009: [16,36,62,89,111,167,230,302],
    2008: [16,39,67,92,118,170,233,305],
}

CONE_SIZE_PAC = {
    2022: [16,25,38,51,65,79,93,120,146],
    2021: [16,25,37,51,64,77,89,114,138],
    2020: [16,25,38,51,65,78,91,115,138],
    2019: [16,25,38,48,62,88,115,145],
    2018: [16,25,39,50,66,94,125,162],
    2017: [16,25,40,51,66,93,116,151],
    2016: [16,27,42,55,70,100,137,172],
    2015: [16,26,42,54,69,100,143,182],
    2014: [16,30,46,62,79,105,154,190],
    2013: [16,30,49,66,82,111,157,197],
    2012: [16,33,52,72,89,121,170,216],
    2011: [16,33,59,79,98,134,187,230],
    2010: [16,36,59,82,102,138,174,220],
    2009: [16,36,59,85,105,148,187,230],
    2008: [16,36,66,92,115,161,210,256],
}