---
hide:
  - toc
---

# Overview

| Operation | Description | Available on |
| - | - | - |
| **Input / Output** | | |
| `read_file` | Read OGR supported file format into a GeoDataFrame. | [`root`][polars_st.read_file] |
| `write_file` | Write the GeoDataFrame to an OGR supported file format. | [`DataFrame`][polars_st.GeoDataFrameNameSpace.write_file] |
| `write_geojson` | Serialize to GeoJSON FeatureCollection representation. | [`DataFrame`][polars_st.GeoDataFrameNameSpace.write_geojson] |
| `write_ndgeojson` | Serialize to newline-delimited GeoJSON FeatureCollection representation. | [`DataFrame`][polars_st.GeoDataFrameNameSpace.write_ndgeojson] |
| **Parsing** | | |
| `from_wkb` | Parse geometries from Well-Known Binary (WKB) representation | [`root`][polars_st.from_wkb] |
| `from_wkt` | Parse geometries from Well-Known Text (WKT) representation | [`root`][polars_st.from_wkt] |
| `from_ewkt` | Parse geometries from Extended Well-Known Text (EWKT) representation | [`root`][polars_st.from_ewkt] |
| `from_geojson` | Parse geometries from GeoJSON representation. | [`root`][polars_st.from_geojson] |
| `from_xy` | Create points from x, y (, z) coordinates | [`root`][polars_st.from_xy] |
| `from_shapely` | Parse geometries from shapely objects | [`root`][polars_st.from_shapely] |
| `from_geopandas` | Create `GeoDataFrame` or `GeoSeries` from Geopandas equivalent. | [`root`][polars_st.from_geopandas] |
| **Serialization** | | |
| `to_wkt` | Serialize each geometry as WKT (Well-Known Text). | [`root`][polars_st.to_wkt], [`Expr`][polars_st.GeoExprNameSpace.to_wkt], [`Series`][polars_st.GeoSeriesNameSpace.to_wkt], [`DataFrame`][polars_st.GeoDataFrameNameSpace.to_wkt] |
| `to_ewkt` | Serialize each geometry as EWKT (Extended Well-Known Text). | [`root`][polars_st.to_ewkt], [`Expr`][polars_st.GeoExprNameSpace.to_ewkt], [`Series`][polars_st.GeoSeriesNameSpace.to_ewkt], [`DataFrame`][polars_st.GeoDataFrameNameSpace.to_ewkt] |
| `to_wkb` | Serialize each geometry as WKB (Well-Known Binary). | [`root`][polars_st.to_wkb], [`Expr`][polars_st.GeoExprNameSpace.to_wkb], [`Series`][polars_st.GeoSeriesNameSpace.to_wkb], [`DataFrame`][polars_st.GeoDataFrameNameSpace.to_wkb] | 
| `to_geojson` | Serialize each geometry as GeoJSON. | [`root`][polars_st.to_geojson], [`Expr`][polars_st.GeoExprNameSpace.to_geojson], [`Series`][polars_st.GeoSeriesNameSpace.to_geojson], [`DataFrame`][polars_st.GeoDataFrameNameSpace.to_geojson] |
| `to_dict` | Convert each geometry to a GeoJSON-like  Python [`dict`][] object. | [`root`][polars_st.to_dict], [`Expr`][polars_st.GeoExprNameSpace.to_dict], [`Series`][polars_st.GeoSeriesNameSpace.to_dict], [`DataFrame`][polars_st.GeoDataFrameNameSpace.to_dict] |
| `to_shapely` | Convert each geometry to a Shapely object. | [`root`][polars_st.to_shapely], [`Expr`][polars_st.GeoExprNameSpace.to_shapely], [`Series`][polars_st.GeoSeriesNameSpace.to_shapely], [`DataFrame`][polars_st.GeoDataFrameNameSpace.to_shapely] |
| `to_geopandas` | Convert DataFrame or Series to GeoPandas equivalent. | [`Series`][polars_st.GeoSeriesNameSpace.to_geopandas], [`DataFrame`][polars_st.GeoDataFrameNameSpace.to_geopandas] |
| `__geo_interface__` | Return a [`dict`][] representation of a `GeometryCollection` or `FeatureCollection` | [`Series`][polars_st.GeoSeriesNameSpace.__geo_interface__], [`DataFrame`][polars_st.GeoDataFrameNameSpace.__geo_interface__] |
| **General operations** | | |
| `geometry_type` | Return the type ID of each geometry.. | [`root`][polars_st.geometry_type], [`Expr`][polars_st.GeoExprNameSpace.geometry_type], [`Series`][polars_st.GeoSeriesNameSpace.geometry_type] |
| `dimensions` | Return the inherent dimensionality of each geometry.. | [`root`][polars_st.dimensions], [`Expr`][polars_st.GeoExprNameSpace.dimensions], [`Series`][polars_st.GeoSeriesNameSpace.dimensions] |
| `coordinate_dimension` | Return the coordinate dimension of each geometry.. | [`root`][polars_st.coordinate_dimension], [`Expr`][polars_st.GeoExprNameSpace.coordinate_dimension], [`Series`][polars_st.GeoSeriesNameSpace.coordinate_dimension] |
| `area` | Return the area of each geometry. | [`root`][polars_st.area], [`Expr`][polars_st.GeoExprNameSpace.area], [`Series`][polars_st.GeoSeriesNameSpace.area] |
| `bounds` | Return the bounds of each geometry. | [`root`][polars_st.bounds], [`Expr`][polars_st.GeoExprNameSpace.bounds], [`Series`][polars_st.GeoSeriesNameSpace.bounds] |
| `length` | Return the length of each geometry. | [`root`][polars_st.length], [`Expr`][polars_st.GeoExprNameSpace.length], [`Series`][polars_st.GeoSeriesNameSpace.length] |
| `minimum_clearance` | Return the minimum clearance of each geometry.. | [`root`][polars_st.minimum_clearance], [`Expr`][polars_st.GeoExprNameSpace.minimum_clearance], [`Series`][polars_st.GeoSeriesNameSpace.minimum_clearance] |
| `x` | Return the `x` value of Point geometries. | [`root`][polars_st.x], [`Expr`][polars_st.GeoExprNameSpace.x], [`Series`][polars_st.GeoSeriesNameSpace.x] |
| `y` | Return the `y` value of Point geometries. | [`root`][polars_st.y], [`Expr`][polars_st.GeoExprNameSpace.y], [`Series`][polars_st.GeoSeriesNameSpace.y] |
| `z` | Return the `z` value of Point geometries. | [`root`][polars_st.z], [`Expr`][polars_st.GeoExprNameSpace.z], [`Series`][polars_st.GeoSeriesNameSpace.z] |
| `m` | Return the `m` value of Point geometries. | [`root`][polars_st.m], [`Expr`][polars_st.GeoExprNameSpace.m], [`Series`][polars_st.GeoSeriesNameSpace.m] |
| `count_coordinates` | Return the number of coordinates in each geometry. | [`root`][polars_st.count_coordinates], [`Expr`][polars_st.GeoExprNameSpace.count_coordinates], [`Series`][polars_st.GeoSeriesNameSpace.count_coordinates] |
| `coordinates` | Return the coordinates of each geometry. | [`root`][polars_st.coordinates], [`Expr`][polars_st.GeoExprNameSpace.coordinates], [`Series`][polars_st.GeoSeriesNameSpace.coordinates] |
| `count_geometries` | Return the number of parts in multipart geometries. | [`root`][polars_st.count_geometries], [`Expr`][polars_st.GeoExprNameSpace.count_geometries], [`Series`][polars_st.GeoSeriesNameSpace.count_geometries] |
| `get_geometry` | Return the nth part of multipart geometries. | [`root`][polars_st.get_geometry], [`Expr`][polars_st.GeoExprNameSpace.get_geometry], [`Series`][polars_st.GeoSeriesNameSpace.get_geometry] |
| `count_points` | Return the number of points in LineString geometries. | [`root`][polars_st.count_points], [`Expr`][polars_st.GeoExprNameSpace.count_points], [`Series`][polars_st.GeoSeriesNameSpace.count_points] |
| `get_point` | Return the nth point of LineString geometries. | [`root`][polars_st.get_point], [`Expr`][polars_st.GeoExprNameSpace.get_point], [`Series`][polars_st.GeoSeriesNameSpace.get_point] |
| `count_interior_rings` | Return the number of interior rings in Polygon geometries. | [`root`][polars_st.count_interior_rings], [`Expr`][polars_st.GeoExprNameSpace.count_interior_rings], [`Series`][polars_st.GeoSeriesNameSpace.count_interior_rings] |
| `get_interior_ring` | Return the nth ring of Polygon geometries. | [`root`][polars_st.get_interior_ring], [`Expr`][polars_st.GeoExprNameSpace.get_interior_ring], [`Series`][polars_st.GeoSeriesNameSpace.get_interior_ring] |
| `exterior_ring` | Return the exterior ring of Polygon geometries. | [`root`][polars_st.exterior_ring], [`Expr`][polars_st.GeoExprNameSpace.exterior_ring], [`Series`][polars_st.GeoSeriesNameSpace.exterior_ring] |
| `interior_rings` | Return the list of interior rings for Polygon geometries. | [`root`][polars_st.interior_rings], [`Expr`][polars_st.GeoExprNameSpace.interior_rings], [`Series`][polars_st.GeoSeriesNameSpace.interior_rings] |
| `parts` | Return a list of parts for multipart geometries. | [`root`][polars_st.parts], [`Expr`][polars_st.GeoExprNameSpace.parts], [`Series`][polars_st.GeoSeriesNameSpace.parts] |
| `precision` | Return the precision of each geometry. | [`root`][polars_st.precision], [`Expr`][polars_st.GeoExprNameSpace.precision], [`Series`][polars_st.GeoSeriesNameSpace.precision] |
| `set_precision` | Set the precision of each geometry to a certain grid size. | [`root`][polars_st.set_precision], [`Expr`][polars_st.GeoExprNameSpace.set_precision], [`Series`][polars_st.GeoSeriesNameSpace.set_precision] |
| `distance` | Return the distance of each geometry to other. | [`Expr`][polars_st.GeoExprNameSpace.distance], [`Series`][polars_st.GeoSeriesNameSpace.distance] |
| `hausdorff_distance` | Return the geometries hausdorff distance to other geometries. | [`Expr`][polars_st.GeoExprNameSpace.hausdorff_distance], [`Series`][polars_st.GeoSeriesNameSpace.hausdorff_distance] |
| `frechet_distance` | Return the geometries frechet distance to other geometries. | [`Expr`][polars_st.GeoExprNameSpace.frechet_distance], [`Series`][polars_st.GeoSeriesNameSpace.frechet_distance] |
| **Projection operations** | | |
| `srid` | Return the SRID of each geometry. | [`root`][polars_st.srid], [`Expr`][polars_st.GeoExprNameSpace.srid], [`Series`][polars_st.GeoSeriesNameSpace.srid] |
| `set_srid` | Set the SRID of each geometry to a given value. | [`root`][polars_st.set_srid], [`Expr`][polars_st.GeoExprNameSpace.set_srid], [`Series`][polars_st.GeoSeriesNameSpace.set_srid] |
| `to_srid` | Transform the coordinates of each geometry into a new CRS. | [`root`][polars_st.to_srid], [`Expr`][polars_st.GeoExprNameSpace.to_srid], [`Series`][polars_st.GeoSeriesNameSpace.to_srid] |
| **Unary predicates** | | |
| `has_z` | Return `True` for geometries that has `z` coordinate values. | [`root`][polars_st.has_z], [`Expr`][polars_st.GeoExprNameSpace.has_z], [`Series`][polars_st.GeoSeriesNameSpace.has_z] |
| `has_m` | Return `True` for geometries that has `m` coordinate values. | [`root`][polars_st.has_m], [`Expr`][polars_st.GeoExprNameSpace.has_m], [`Series`][polars_st.GeoSeriesNameSpace.has_m] |
| `is_ccw` | Return `True` for linear geometries with counter-clockwise coord sequence. | [`root`][polars_st.is_ccw], [`Expr`][polars_st.GeoExprNameSpace.is_ccw], [`Series`][polars_st.GeoSeriesNameSpace.is_ccw] |
| `is_closed` | Return `True` for closed linear geometries. | [`root`][polars_st.is_closed], [`Expr`][polars_st.GeoExprNameSpace.is_closed], [`Series`][polars_st.GeoSeriesNameSpace.is_closed] |
| `is_empty` | Return `True` for empty geometries. | [`root`][polars_st.is_empty], [`Expr`][polars_st.GeoExprNameSpace.is_empty], [`Series`][polars_st.GeoSeriesNameSpace.is_empty] |
| `is_ring` | Return `True` for ring geometries. | [`root`][polars_st.is_ring], [`Expr`][polars_st.GeoExprNameSpace.is_ring], [`Series`][polars_st.GeoSeriesNameSpace.is_ring] |
| `is_simple` | Return `True` for simple geometries. | [`root`][polars_st.is_simple], [`Expr`][polars_st.GeoExprNameSpace.is_simple], [`Series`][polars_st.GeoSeriesNameSpace.is_simple] |
| `is_valid` | Return `True` for valid geometries. | [`root`][polars_st.is_valid], [`Expr`][polars_st.GeoExprNameSpace.is_valid], [`Series`][polars_st.GeoSeriesNameSpace.is_valid] |
| `is_valid_reason` | Return an explanation string for the invalidity of each geometry. | [`root`][polars_st.is_valid_reason], [`Expr`][polars_st.GeoExprNameSpace.is_valid_reason], [`Series`][polars_st.GeoSeriesNameSpace.is_valid_reason] |
| **Binary predicates** | | |
| `crosses` | Return `True` when each geometry crosses other. | [`Expr`][polars_st.GeoExprNameSpace.crosses], [`Series`][polars_st.GeoSeriesNameSpace.crosses] |
| `contains` | Return `True` when each geometry contains other. | [`Expr`][polars_st.GeoExprNameSpace.contains], [`Series`][polars_st.GeoSeriesNameSpace.contains] |
| `contains_properly` | Return `True` when each geometry properly crosses other. | [`Expr`][polars_st.GeoExprNameSpace.contains_properly], [`Series`][polars_st.GeoSeriesNameSpace.contains_properly] |
| `covered_by` | Return `True` when each geometry is covered by other. | [`Expr`][polars_st.GeoExprNameSpace.covered_by], [`Series`][polars_st.GeoSeriesNameSpace.covered_by] |
| `covers` | Return `True` when each geometry covers other. | [`Expr`][polars_st.GeoExprNameSpace.covers], [`Series`][polars_st.GeoSeriesNameSpace.covers] |
| `disjoint` | Return `True` when each geometry is disjoint from other. | [`Expr`][polars_st.GeoExprNameSpace.disjoint], [`Series`][polars_st.GeoSeriesNameSpace.disjoint] |
| `dwithin` | Return `True` when each geometry is within given distance to other. | [`Expr`][polars_st.GeoExprNameSpace.dwithin], [`Series`][polars_st.GeoSeriesNameSpace.dwithin] |
| `intersects` | Return `True` when each geometry intersects other. | [`Expr`][polars_st.GeoExprNameSpace.intersects], [`Series`][polars_st.GeoSeriesNameSpace.intersects] |
| `overlaps` |Return `True` when each geometry overlaps other. | [`Expr`][polars_st.GeoExprNameSpace.overlaps], [`Series`][polars_st.GeoSeriesNameSpace.overlaps] |
| `touches` |Return `True` when each geometry touches other. | [`Expr`][polars_st.GeoExprNameSpace.touches], [`Series`][polars_st.GeoSeriesNameSpace.touches] |
| `within` |Return `True` when each geometry is within other. | [`Expr`][polars_st.GeoExprNameSpace.within], [`Series`][polars_st.GeoSeriesNameSpace.within] |
| `equals` | Return `True` when each geometry is equal to other. | [`Expr`][polars_st.GeoExprNameSpace.equals], [`Series`][polars_st.GeoSeriesNameSpace.equals] |
| `equals_exact` | Return `True` when each geometry is equal to other. | [`Expr`][polars_st.GeoExprNameSpace.equals_exact], [`Series`][polars_st.GeoSeriesNameSpace.equals_exact] |
| `equals_identical` | Return `True` when each geometry is equal to other. | [`Expr`][polars_st.GeoExprNameSpace.equals_identical], [`Series`][polars_st.GeoSeriesNameSpace.equals_identical] |
| `relate` | Return the DE-9IM intersection matrix of each geometry with other. | [`Expr`][polars_st.GeoExprNameSpace.relate], [`Series`][polars_st.GeoSeriesNameSpace.relate] |
| `relate_pattern` | Return `True` when the DE-9IM intersection matrix matches a given pattern. | [`Expr`][polars_st.GeoExprNameSpace.relate_pattern], [`Series`][polars_st.GeoSeriesNameSpace.relate_pattern] |
| **Set operations** | | |
| `union` | Return the union of each geometry with other. | [`Expr`][polars_st.GeoExprNameSpace.union], [`Series`][polars_st.GeoSeriesNameSpace.union] |
| `unary_union` | Return the unary union of each geometry. | [`Expr`][polars_st.GeoExprNameSpace.unary_union], [`Series`][polars_st.GeoSeriesNameSpace.unary_union] |
| `coverage_union` | Return the coverage union of each geometry with other. | [`Expr`][polars_st.GeoExprNameSpace.coverage_union], [`Series`][polars_st.GeoSeriesNameSpace.coverage_union] |
| `intersection` | Return the intersection of each geometry with other. | [`Expr`][polars_st.GeoExprNameSpace.intersection], [`Series`][polars_st.GeoSeriesNameSpace.intersection] |
| `difference` | Return the difference of each geometry with other. | [`Expr`][polars_st.GeoExprNameSpace.difference], [`Series`][polars_st.GeoSeriesNameSpace.difference] |
| `symmetric_difference` | Return the symmetric difference of each geometry with other. | [`Expr`][polars_st.GeoExprNameSpace.symmetric_difference], [`Series`][polars_st.GeoSeriesNameSpace.symmetric_difference] |
| **Constructive operations** | | |
| `boundary` | Return the topological boundary of each geometry. | [`root`][polars_st.boundary], [`Expr`][polars_st.GeoExprNameSpace.boundary], [`Series`][polars_st.GeoSeriesNameSpace.boundary] |
| `buffer` | Return a buffer around each geometry. | [`root`][polars_st.buffer], [`Expr`][polars_st.GeoExprNameSpace.buffer], [`Series`][polars_st.GeoSeriesNameSpace.buffer] |
| `offset_curve` | Return a line at a given distance of each geometry. | [`root`][polars_st.offset_curve], [`Expr`][polars_st.GeoExprNameSpace.offset_curve], [`Series`][polars_st.GeoSeriesNameSpace.offset_curve] |
| `centroid` | Return the centroid of each geometry. | [`root`][polars_st.centroid], [`Expr`][polars_st.GeoExprNameSpace.centroid], [`Series`][polars_st.GeoSeriesNameSpace.centroid] |
| `center` | Return the center of each geometry. | [`root`][polars_st.center], [`Expr`][polars_st.GeoExprNameSpace.center], [`Series`][polars_st.GeoSeriesNameSpace.center] |
| `clip_by_rect` | Clips each geometry by a bounding rectangle. | [`root`][polars_st.clip_by_rect], [`Expr`][polars_st.GeoExprNameSpace.clip_by_rect], [`Series`][polars_st.GeoSeriesNameSpace.clip_by_rect] |
| `convex_hull` | Return the convex hull of each geometry. | [`root`][polars_st.convex_hull], [`Expr`][polars_st.GeoExprNameSpace.convex_hull], [`Series`][polars_st.GeoSeriesNameSpace.convex_hull] |
| `concave_hull` | Return the concave hull of each geometry. | [`root`][polars_st.concave_hull], [`Expr`][polars_st.GeoExprNameSpace.concave_hull], [`Series`][polars_st.GeoSeriesNameSpace.concave_hull] |
| `segmentize` | | [`root`][polars_st.segmentize], [`Expr`][polars_st.GeoExprNameSpace.segmentize], [`Series`][polars_st.GeoSeriesNameSpace.segmentize] |
| `envelope` | Return the envelope of each geometry. | [`root`][polars_st.envelope], [`Expr`][polars_st.GeoExprNameSpace.envelope], [`Series`][polars_st.GeoSeriesNameSpace.envelope] |
| `extract_unique_points` | | [`root`][polars_st.extract_unique_points], [`Expr`][polars_st.GeoExprNameSpace.extract_unique_points], [`Series`][polars_st.GeoSeriesNameSpace.extract_unique_points] |
| `build_area` | | [`root`][polars_st.build_area], [`Expr`][polars_st.GeoExprNameSpace.build_area], [`Series`][polars_st.GeoSeriesNameSpace.build_area] |
| `make_valid` | | [`root`][polars_st.make_valid], [`Expr`][polars_st.GeoExprNameSpace.make_valid], [`Series`][polars_st.GeoSeriesNameSpace.make_valid] |
| `normalize` | | [`root`][polars_st.normalize], [`Expr`][polars_st.GeoExprNameSpace.normalize], [`Series`][polars_st.GeoSeriesNameSpace.normalize] |
| `node` | | [`root`][polars_st.node], [`Expr`][polars_st.GeoExprNameSpace.node], [`Series`][polars_st.GeoSeriesNameSpace.node] |
| `point_on_surface` | Return a point that intersects each geometry. | [`root`][polars_st.point_on_surface], [`Expr`][polars_st.GeoExprNameSpace.point_on_surface], [`Series`][polars_st.GeoSeriesNameSpace.point_on_surface] |
| `remove_repeated_points` | Remove the repeated points for each geometry. | [`root`][polars_st.remove_repeated_points], [`Expr`][polars_st.GeoExprNameSpace.remove_repeated_points], [`Series`][polars_st.GeoSeriesNameSpace.remove_repeated_points] |
| `reverse` | Reverse the coordinates order of each geometry. | [`root`][polars_st.reverse], [`Expr`][polars_st.GeoExprNameSpace.reverse], [`Series`][polars_st.GeoSeriesNameSpace.reverse] |
| `simplify` | Simplify each geometry with a given tolerance. | [`root`][polars_st.simplify], [`Expr`][polars_st.GeoExprNameSpace.simplify], [`Series`][polars_st.GeoSeriesNameSpace.simplify] |
| `force_2d` | Force the dimensionality of a geometry to 2D. | [`root`][polars_st.force_2d], [`Expr`][polars_st.GeoExprNameSpace.force_2d], [`Series`][polars_st.GeoSeriesNameSpace.force_2d] |
| `force_3d` | Force the dimensionality of a geometry to 3D. | [`root`][polars_st.force_3d], [`Expr`][polars_st.GeoExprNameSpace.force_3d], [`Series`][polars_st.GeoSeriesNameSpace.force_3d] |
| `flip_coordinates` | Flip the x and y coordinates of each geometry. | [`root`][polars_st.flip_coordinates], [`Expr`][polars_st.GeoExprNameSpace.flip_coordinates], [`Series`][polars_st.GeoSeriesNameSpace.flip_coordinates] |
| `minimum_rotated_rectangle` | | [`root`][polars_st.minimum_rotated_rectangle], [`Expr`][polars_st.GeoExprNameSpace.minimum_rotated_rectangle], [`Series`][polars_st.GeoSeriesNameSpace.minimum_rotated_rectangle] | |
| `snap` | | [`Expr`][polars_st.GeoExprNameSpace.snap], [`Series`][polars_st.GeoSeriesNameSpace.snap] |
| `shortest_line` | Return the shortest line between each geometry and other. | [`Expr`][polars_st.GeoExprNameSpace.shortest_line], [`Series`][polars_st.GeoSeriesNameSpace.shortest_line] |
| `sjoin` | Perform a spatial join operation with another DataFrame. | [`DataFrame`][polars_st.GeoDataFrameNameSpace.sjoin], [`LazyFrame`][polars_st.GeoLazyFrameNameSpace.sjoin] |
| **Affine transforms** | | |
| `affine_transform` | | [`root`][polars_st.affine_transform], [`Expr`][polars_st.GeoExprNameSpace.affine_transform], [`Series`][polars_st.GeoSeriesNameSpace.affine_transform] |
| `translate` | | [`root`][polars_st.translate], [`Expr`][polars_st.GeoExprNameSpace.translate], [`Series`][polars_st.GeoSeriesNameSpace.translate] |
| `rotate` | | [`root`][polars_st.rotate], [`Expr`][polars_st.GeoExprNameSpace.rotate], [`Series`][polars_st.GeoSeriesNameSpace.rotate] |
| `scale` | | [`root`][polars_st.scale], [`Expr`][polars_st.GeoExprNameSpace.scale], [`Series`][polars_st.GeoSeriesNameSpace.scale] |
| **LineString operations** | | |
| `interpolate` | | [`root`][polars_st.interpolate], [`Expr`][polars_st.GeoExprNameSpace.interpolate], [`Series`][polars_st.GeoSeriesNameSpace.interpolate] |
| `project` | | [`Expr`][polars_st.GeoExprNameSpace.project], [`Series`][polars_st.GeoSeriesNameSpace.project] |
| `line_merge` | | [`root`][polars_st.line_merge], [`Expr`][polars_st.GeoExprNameSpace.line_merge], [`Series`][polars_st.GeoSeriesNameSpace.line_merge] |
| `shared_paths` | | [`Expr`][polars_st.GeoExprNameSpace.shared_paths], [`Series`][polars_st.GeoSeriesNameSpace.shared_paths] |
| **Aggregation** | | |
| `total_bounds` | Return the total bounds of all geometries. | [`root`][polars_st.total_bounds], [`Expr`][polars_st.GeoExprNameSpace.total_bounds], [`Series`][polars_st.GeoSeriesNameSpace.total_bounds] |
| `multipoint` | Aggregate Point geometries into a single MultiPoint. | [`root`][polars_st.multipoint], [`Expr`][polars_st.GeoExprNameSpace.multipoint], [`Series`][polars_st.GeoSeriesNameSpace.multipoint] |
| `multilinestring` | Aggregate LineString geometries into a single MultiLineString. | [`root`][polars_st.multilinestring], [`Expr`][polars_st.GeoExprNameSpace.multilinestring], [`Series`][polars_st.GeoSeriesNameSpace.multilinestring] |
| `multipolygon` | Aggregate Polygon geometries into a single MultiPolygon. | [`root`][polars_st.multipolygon], [`Expr`][polars_st.GeoExprNameSpace.multipolygon], [`Series`][polars_st.GeoSeriesNameSpace.multipolygon] |
| `geometrycollection` | Aggregate geometries into a single GeometryCollection. | [`root`][polars_st.geometrycollection], [`Expr`][polars_st.GeoExprNameSpace.geometrycollection], [`Series`][polars_st.GeoSeriesNameSpace.geometrycollection] |
| `collect` | Aggregate geometries into a single MultiPart geometry or GeometryCollection. | [`root`][polars_st.collect], [`Expr`][polars_st.GeoExprNameSpace.collect], [`Series`][polars_st.GeoSeriesNameSpace.collect] |
| `union_all` | Return the union of all geometries. | [`root`][polars_st.union_all], [`Expr`][polars_st.GeoExprNameSpace.union_all], [`Series`][polars_st.GeoSeriesNameSpace.union_all] |
| `coverage_union_all` | Return the coverage union of all geometries. | [`root`][polars_st.coverage_union_all], [`Expr`][polars_st.GeoExprNameSpace.coverage_union_all], [`Series`][polars_st.GeoSeriesNameSpace.coverage_union_all] |
| `intersection_all` | Return the intersection of all geometries. | [`root`][polars_st.intersection_all], [`Expr`][polars_st.GeoExprNameSpace.intersection_all], [`Series`][polars_st.GeoSeriesNameSpace.intersection_all] |
| `difference_all` | Return the difference of all geometries | [`root`][polars_st.difference_all], [`Expr`][polars_st.GeoExprNameSpace.difference_all], [`Series`][polars_st.GeoSeriesNameSpace.difference_all] |
| `symmetric_difference_all` | Return the symmetric difference of all geometries. | [`root`][polars_st.symmetric_difference_all], [`Expr`][polars_st.GeoExprNameSpace.symmetric_difference_all], [`Series`][polars_st.GeoSeriesNameSpace.symmetric_difference_all] |
| `polygonize` | | [`root`][polars_st.polygonize], [`Expr`][polars_st.GeoExprNameSpace.polygonize], [`Series`][polars_st.GeoSeriesNameSpace.polygonize] |
| `voronoi_polygons` | Return a Voronoi diagram of all geometries vertices. | [`root`][polars_st.voronoi_polygons], [`Expr`][polars_st.GeoExprNameSpace.voronoi_polygons], [`Series`][polars_st.GeoSeriesNameSpace.voronoi_polygons] |
| `delaunay_triangles` | Return a Delaunay triangulation of all geometries vertices. | [`root`][polars_st.delaunay_triangles], [`Expr`][polars_st.GeoExprNameSpace.delaunay_triangles], [`Series`][polars_st.GeoSeriesNameSpace.delaunay_triangles] |
| **Plotting** | | |
| `plot` | Create a map plot of a GeoSeries or GeoDataFrame. | [`Series`][polars_st.GeoSeriesNameSpace.plot], [`DataFrame`][polars_st.GeoDataFrameNameSpace.plot] |
