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
