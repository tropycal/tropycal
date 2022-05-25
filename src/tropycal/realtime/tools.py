r"""
This is a modified version of Cartopy's shapereader functionality, specified to directly read in an already-existing
Shapely object as opposed to expecting a local shapefile to be read in.
"""
import shapely.geometry as sgeom
import shapefile

class Record:
    """
    A single logical entry from a shapefile, combining the attributes with
    their associated geometry.

    """
    def __init__(self, shape, attributes, fields):
        self._shape = shape

        self._bounds = None
        # if the record defines a bbox, then use that for the shape's bounds,
        # rather than using the full geometry in the bounds property
        if hasattr(shape, 'bbox'):
            self._bounds = tuple(shape.bbox)

        self._geometry = None
        """The cached geometry instance for this Record."""

        self.attributes = attributes
        """A dictionary mapping attribute names to attribute values."""

        self._fields = fields

    def __repr__(self):
        return f'<Record: {self.geometry!r}, {self.attributes!r}, <fields>>'

    def __str__(self):
        return f'Record({self.geometry}, {self.attributes}, <fields>)'

    @property
    def bounds(self):
        """
        The bounds of this Record's :meth:`~Record.geometry`.

        """
        if self._bounds is None:
            self._bounds = self.geometry.bounds
        return self._bounds

    @property
    def geometry(self):
        """
        A shapely.geometry instance for this Record.

        The geometry may be ``None`` if a null shape is defined in the
        shapefile.

        """
        if not self._geometry and self._shape.shapeType != shapefile.NULL:
            self._geometry = sgeom.shape(self._shape)
        return self._geometry

class BasicReader:
    """
    Provide an interface for accessing the contents of a shapefile.

    The primary methods used on a Reader instance are
    :meth:`~Reader.records` and :meth:`~Reader.geometries`.

    """
    def __init__(self, reader):
        # Validate the filename/shapefile
        self._reader = reader
        if reader.shp is None or reader.shx is None or reader.dbf is None:
            raise ValueError("Incomplete shapefile definition "
                             "in '%s'." % filename)

        self._fields = self._reader.fields


    def close(self):
        return self._reader.close()

    def __len__(self):
        return len(self._reader)

    def geometries(self):
        """
        Return an iterator of shapely geometries from the shapefile.

        This interface is useful for accessing the geometries of the
        shapefile where knowledge of the associated metadata is not necessary.
        In the case where further metadata is needed use the
        :meth:`~Reader.records`
        interface instead, extracting the geometry from the record with the
        :meth:`~Record.geometry` method.

        """
        to_return = []
        for shape in self._reader.iterShapes():
            # Skip the shape that can not be represented as geometry.
            if shape.shapeType != shapefile.NULL:
                to_return.append(sgeom.shape(shape))
        return to_return

    def records(self):
        """
        Return an iterator of :class:`~Record` instances.

        """
        # Ignore the "DeletionFlag" field which always comes first
        to_return = []
        fields = self._reader.fields[1:]
        for shape_record in self._reader.iterShapeRecords():
            attributes = shape_record.record.as_dict()
            to_return.append(Record(shape_record.shape, attributes, fields))
        return to_return
