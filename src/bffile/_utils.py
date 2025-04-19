from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple
from xml.etree import ElementTree as ET

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ome_types import OME


log = logging.getLogger("bffile")


class PhysicalPixelSizes(NamedTuple):
    """NamedTuple with physical pixel sizes."""

    z: float | None
    y: float | None
    x: float | None

    def __repr__(self) -> str:
        return f"PhysicalPixelSizes(z={self.z}, y={self.y}, x={self.x})"


class OMEShape(NamedTuple):
    """NamedTuple with OME metadata shape."""

    t: int
    c: int
    z: int
    y: int
    x: int
    rgb: int

    def __repr__(self) -> str:
        return f"TCZXYrgb({self.t}, {self.c}, {self.z}, {self.y}, {self.x}, {self.rgb})"


class CoreMeta(NamedTuple):
    """NamedTuple with core bioformats metadata. (not OME meta)."""

    shape: OMEShape
    dtype: np.dtype
    series_count: int
    is_rgb: bool
    is_interleaved: bool
    dimension_order: str
    resolution_count: int


class DimensionNames:
    Time = "T"
    Channel = "C"
    SpatialZ = "Z"
    SpatialY = "Y"
    SpatialX = "X"
    Samples = "S"
    MosaicTile = "M"


def get_coords_from_ome(
    ome: OME, scene_index: int
) -> Mapping[str, list[Any] | np.ndarray | Any]:
    """
    Process the OME metadata to retrieve the coordinate planes.

    Parameters
    ----------
    ome: OME
        A constructed OME object to retrieve data from.
    scene_index: int
        The current operating scene index to pull metadata from.

    Returns
    -------
    coords: Dict[str, Union[List[Any], Union[npt.ArrayLike, Any]]]
        The coordinate planes / data for each dimension.
    """
    # Select scene
    scene_meta = ome.images[scene_index]

    # Get coordinate planes
    coords: dict[str, list[str] | np.ndarray] = {}

    # Channels
    # Channel name isn't required by OME spec, so try to use it but
    # roll back to ID if not found
    coords[DimensionNames.Channel] = [
        channel.name if channel.name is not None else channel.id
        for channel in scene_meta.pixels.channels
    ]

    # Time
    # If global linear timescale we can np.linspace with metadata
    if scene_meta.pixels.time_increment is not None:
        coords[DimensionNames.Time] = generate_coord_array(
            0, scene_meta.pixels.size_t, scene_meta.pixels.time_increment
        )
    # If non global linear timescale, we need to create an array of every plane
    # time value
    elif scene_meta.pixels.size_t > 1:
        if len(scene_meta.pixels.planes) > 0:
            t_index_to_delta_map = {
                p.the_t: p.delta_t or 0 for p in scene_meta.pixels.planes
            }
            coords[DimensionNames.Time] = np.asarray(t_index_to_delta_map.values())
        else:
            coords[DimensionNames.Time] = np.linspace(
                0,
                scene_meta.pixels.size_t - 1,
                scene_meta.pixels.size_t,
            )

    # Handle Spatial Dimensions
    if scene_meta.pixels.physical_size_z is not None:
        coords[DimensionNames.SpatialZ] = generate_coord_array(
            0, scene_meta.pixels.size_z, scene_meta.pixels.physical_size_z
        )
    if scene_meta.pixels.physical_size_y is not None:
        coords[DimensionNames.SpatialY] = generate_coord_array(
            0, scene_meta.pixels.size_y, scene_meta.pixels.physical_size_y
        )
    if scene_meta.pixels.physical_size_x is not None:
        coords[DimensionNames.SpatialX] = generate_coord_array(
            0, scene_meta.pixels.size_x, scene_meta.pixels.physical_size_x
        )

    return coords


def clean_ome_xml_for_known_issues(xml: str) -> str:
    """
    Clean an OME XML string for known issues created by AICS or MicroManager
    systems and tools.

    Commonly this is used for cleaning a file produced by AICS prior to noticing the
    issue (2021), or for other users of aicsimageio as a whole prior to 4.x series of
    releases.

    The result of this function should be an OME XML string that is relatively the
    same (no major pieces missing) but that validates against the reference OME
    XSD.

    Parameters
    ----------
    xml: str
        The OME XML string to clean for errors.

    Returns
    -------
    cleaned_xml: str
        The cleaned OME XML string.

    Raises
    ------
    ValueError
        Provided XML does not contain a namespace.
    """
    # Store list of changes to print out with warning
    metadata_changes = []

    # Read in XML
    root = ET.fromstring(xml)

    # Get the namespace
    # In XML etree this looks like
    # "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
    # and must prepend any etree finds
    namespace_matches = re.match(r"\{.*\}", root.tag)
    if namespace_matches is not None:
        namespace = namespace_matches.group(0)
    else:
        raise ValueError("XML does not contain a namespace")

    # Fix MicroManager Instrument and Detector
    ome_instrument_id = ""
    instrument = root.find(f"{namespace}Instrument")
    if instrument is not None:
        instrument_id = instrument.get("ID")
        if instrument_id == "Microscope":
            ome_instrument_id = generate_ome_instrument_id(0)
            instrument.set("ID", ome_instrument_id)
            metadata_changes.append(
                f"Updated attribute 'ID' from '{instrument_id}' to "
                f"'{ome_instrument_id}' on Instrument element."
            )

            for detector_index, detector in enumerate(
                instrument.findall(f"{namespace}Detector")
            ):
                detector_id = detector.get("ID")
                if detector_id is not None:
                    # Create ome detector id if needed
                    ome_detector_id = None
                    if detector_id == "Camera":
                        ome_detector_id = generate_ome_detector_id(detector_index)
                    elif not detector_id.startswith("Detector:"):
                        ome_detector_id = generate_ome_detector_id(detector_id)

                    # Apply ome detector id if replaced
                    if ome_detector_id is not None:
                        detector.set("ID", ome_detector_id)
                        metadata_changes.append(
                            f"Updated attribute 'ID' from '{detector_id}' to "
                            f"'{ome_detector_id}' on Detector element at "
                            f"position {detector_index}."
                        )

    # Find all Image elements and fix IDs and refs to fixed instruments
    # This is for certain for test files of o.urs and ACTK files
    for image_index, image in enumerate(root.findall(f"{namespace}Image")):
        image_id = image.get("ID")
        if image_id is not None:
            found_image_id = image_id

            if not found_image_id.startswith("Image"):
                ome_image_id = generate_ome_image_id(found_image_id)
                image.set("ID", ome_image_id)
                metadata_changes.append(
                    f"Updated attribute 'ID' from '{image_id}' to '{ome_image_id}' "
                    f"on Image element at position {image_index}."
                )

        # Fix MicroManager bad instrument refs
        instrument_ref = image.find(f"{namespace}InstrumentRef")
        if instrument_ref is not None:
            instrument_ref_id = instrument_ref.get("ID")
            if instrument_ref_id == "Microscope" and ome_instrument_id:
                instrument_ref.set("ID", ome_instrument_id)

        # Find all Pixels elements and fix IDs
        for pixels_index, pixels in enumerate(image.findall(f"{namespace}Pixels")):
            pixels_id = pixels.get("ID")
            if pixels_id is not None:
                found_pixels_id = pixels_id

                if not found_pixels_id.startswith("Pixels"):
                    pixels.set("ID", f"Pixels:{found_pixels_id}")
                    metadata_changes.append(
                        f"Updated attribute 'ID' from '{found_pixels_id}' to "
                        f"Pixels:{found_pixels_id}' on Pixels element at "
                        f"position {pixels_index}."
                    )

            # Determine if there is an out-of-order channel / plane elem
            # This is due to OMEXML "add channel" function
            # That added Channels and appropriate Planes to the XML
            # But, placed them in:
            # Channel
            # Plane
            # Plane
            # ...
            # Channel
            # Plane
            # Plane
            #
            # Instead of grouped together:
            # Channel
            # Channel
            # ...
            # Plane
            # Plane
            # ...
            #
            # This effects all CFE files (new and old) but for different reasons
            pixels_children_out_of_order = False
            encountered_something_besides_channel = False
            encountered_plane = False
            for child in pixels:
                if child.tag != f"{namespace}Channel":
                    encountered_something_besides_channel = True
                if child.tag == f"{namespace}Plane":
                    encountered_plane = True
                if (
                    encountered_something_besides_channel
                    and child.tag == f"{namespace}Channel"
                ):
                    pixels_children_out_of_order = True
                    break
                if encountered_plane and child.tag in [
                    f"{namespace}{t}" for t in ["BinData", "TiffData", "MetadataOnly"]
                ]:
                    pixels_children_out_of_order = True
                    break

            # Ensure order of:
            # channels -> bindata | tiffdata | metadataonly -> planes
            if pixels_children_out_of_order:
                # Get all relevant elems
                channels = [deepcopy(c) for c in pixels.findall(f"{namespace}Channel")]
                bin_data = [deepcopy(b) for b in pixels.findall(f"{namespace}BinData")]
                tiff_data = [
                    deepcopy(t) for t in pixels.findall(f"{namespace}TiffData")
                ]
                # There should only be one metadata only element but to standardize
                # list comprehensions later we findall
                metadata_only = [
                    deepcopy(m) for m in pixels.findall(f"{namespace}MetadataOnly")
                ]
                planes = [deepcopy(p) for p in pixels.findall(f"{namespace}Plane")]

                # Old (2018 ish) cell feature explorer files sometimes contain both
                # an empty metadata only element and filled tiffdata elements
                # Since the metadata only elements are empty we can check this and
                # choose the tiff data elements instead
                #
                # First check if there are any metadata only elements
                if len(metadata_only) == 1:
                    # Now check if _one of_ of the other two choices are filled
                    # ^ in Python is XOR
                    if (len(bin_data) > 0) ^ (len(tiff_data) > 0):
                        metadata_children = list(metadata_only[0])
                        # Now check if the metadata only elem has no children
                        if len(metadata_children) == 0:
                            # If so, just "purge" by creating empty list
                            metadata_only = []

                        # If there are children elements
                        # Return XML and let XMLSchema Validation show error
                        else:
                            return xml

                # After cleaning metadata only, validate the normal behaviors of
                # OME schema
                #
                # Validate that there is only one of bindata, tiffdata, or metadata
                if len(bin_data) > 0:
                    if len(tiff_data) == 0 and len(metadata_only) == 0:
                        selected_choice = bin_data
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml
                elif len(tiff_data) > 0:
                    if len(bin_data) == 0 and len(metadata_only) == 0:
                        selected_choice = tiff_data
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml
                elif len(metadata_only) == 1:
                    if len(bin_data) == 0 and len(tiff_data) == 0:
                        selected_choice = metadata_only
                    else:
                        # Return XML and let XMLSchema Validation show error
                        return xml
                else:
                    # Return XML and let XMLSchema Validation show error
                    return xml

                # Remove all children from element to be replaced
                # with ordered elements
                for elem in list(pixels):
                    pixels.remove(elem)

                # Re-attach elements
                for channel in channels:
                    pixels.append(channel)
                for elem in selected_choice:
                    pixels.append(elem)
                for plane in planes:
                    pixels.append(plane)

                metadata_changes.append(
                    f"Reordered children of Pixels element at position {pixels_index}."
                )

    # This is a result of dumping basically all experiment metadata
    # into "StructuredAnnotation" blocks
    #
    # This affects new (2020) Cell Feature Explorer files
    #
    # Because these are structured annotations we don't want to mess with anyones
    # besides the AICS generated bad structured annotations
    aics_anno_removed_count = 0
    sa = root.find(f"{namespace}StructuredAnnotations")
    if sa is not None:
        for xml_anno in sa.findall(f"{namespace}XMLAnnotation"):
            # At least these are namespaced
            if xml_anno.get("Namespace") == "alleninstitute.org/CZIMetadata":
                # Get ID because some elements have annotation refs
                # in both the base Image element and all plane elements
                aics_anno_id = xml_anno.get("ID")
                for image in root.findall(f"{namespace}Image"):
                    for anno_ref in image.findall(f"{namespace}AnnotationRef"):
                        if anno_ref.get("ID") == aics_anno_id:
                            image.remove(anno_ref)

                    # Clean planes
                    if image is not None:
                        found_image = image

                        pixels_planes: ET.Element | None = found_image.find(
                            f"{namespace}Pixels"
                        )
                        if pixels_planes is not None:
                            for plane in pixels_planes.findall(f"{namespace}Plane"):
                                for anno_ref in plane.findall(
                                    f"{namespace}AnnotationRef"
                                ):
                                    if anno_ref.get("ID") == aics_anno_id:
                                        plane.remove(anno_ref)

                # Remove the whole etree
                sa.remove(xml_anno)
                aics_anno_removed_count += 1

    # Log changes
    if aics_anno_removed_count > 0:
        metadata_changes.append(
            f"Removed {aics_anno_removed_count} AICS generated XMLAnnotations."
        )

    # If there are no annotations in StructuredAnnotations, remove it
    if sa is not None:
        if len(list(sa)) == 0:
            root.remove(sa)

    # If any piece of metadata was changed alert and rewrite
    if len(metadata_changes) > 0:
        log.debug("OME metadata was cleaned for known AICSImageIO 3.x OMEXML errors.")
        log.debug(f"Full list of OME cleaning changes: {metadata_changes}")

        # Write out cleaned XML to string
        xml = ET.tostring(
            root,
            encoding="unicode",
            method="xml",
        )

    return xml


def physical_pixel_sizes(ome: OME, scene: int = 0) -> PhysicalPixelSizes:
    """
    Returns
    -------
    sizes: PhysicalPixelSizes
        Using available metadata, the floats representing physical pixel sizes for
        dimensions Z, Y, and X.

    Notes
    -----
    We currently do not handle unit attachment to these values. Please see the file
    metadata for unit information.
    """
    p = ome.images[scene].pixels
    return PhysicalPixelSizes(p.physical_size_z, p.physical_size_y, p.physical_size_x)


def generate_ome_image_id(image_id: str | int) -> str:
    """
    Naively generates the standard OME image ID using a provided ID.

    Parameters
    ----------
    image_id: Union[str, int]
        A string or int representing the ID for an image.
        In the context of the usage of this function, this is usually used with the
        index of the scene / image.

    Returns
    -------
    ome_image_id: str
        The OME standard for image IDs.
    """
    return f"Image:{image_id}"


def generate_ome_channel_id(image_id: str, channel_id: str | int) -> str:
    """
    Naively generates the standard OME channel ID using a provided ID.

    Parameters
    ----------
    image_id: str
        An image id to pull the image specific index from.
        See: `generate_ome_image_id` for more details.
    channel_id: Union[str, int]
        A string or int representing the ID for a channel.
        In the context of the usage of this function, this is usually used with the
        index of the channel.

    Returns
    -------
    ome_channel_id: str
        The OME standard for channel IDs.

    Notes
    -----
    ImageIds are usually: "Image:0", "Image:1", or "Image:N",
    ChannelIds are usually the combination of image index + channel index --
    "Channel:0:0" for the first channel of the first image for example.
    """
    # Remove the prefix 'Image:' to get just the index
    image_index = image_id.replace("Image:", "")
    return f"Channel:{image_index}:{channel_id}"


def generate_ome_instrument_id(instrument_id: str | int) -> str:
    """
    Naively generates the standard OME instrument ID using a provided ID.

    Parameters
    ----------
    instrument_id: Union[str, int]
        A string or int representing the ID for an instrument.

    Returns
    -------
    ome_instrument_id: str
        The OME standard for instrument IDs.
    """
    return f"Instrument:{instrument_id}"


def generate_ome_detector_id(detector_id: str | int) -> str:
    """
    Naively generates the standard OME detector ID using a provided ID.

    Parameters
    ----------
    detector_id: Union[str, int]
        A string or int representing the ID for a detector.

    Returns
    -------
    ome_detector_id: str
        The OME standard for detector IDs.
    """
    return f"Detector:{detector_id}"


def generate_coord_array(
    start: int | float, stop: int | float, step_size: int | float
) -> np.ndarray:
    """
    Generate an np.ndarray for coordinate values.

    Parameters
    ----------
    start: Union[int, float]
        The start value.
    stop: Union[int, float]
        The stop value.
    step_size: Union[int, float]
        How large each step should be.

    Returns
    -------
    coords: np.ndarray
        The coordinate array.

    Notes
    -----
    In general, we have learned that floating point math is hard....
    This block of code used to use `np.arange` with floats as parameters and
    it was causing errors. To solve, we generate the range with ints and then
    multiply by a float across the entire range to get the proper coords.
    See: https://github.com/AllenCellModeling/aicsimageio/issues/249
    """
    return np.arange(start, stop) * step_size


def _chunk_by_tile_size(n_px: int, tile_length: int) -> tuple[int, ...]:
    n_splits = n_px / tile_length
    n_full_tiles = np.floor(n_splits)

    if n_splits.is_integer():
        tile_chunks = (int(tile_length),) * int(n_full_tiles)
    else:
        edge_tile = n_px - (n_full_tiles * tile_length)
        tile_chunks = (int(tile_length),) * int(n_full_tiles) + (int(edge_tile),)
    return tile_chunks


def get_dask_tile_chunks(
    nt: int, nc: int, nz: int, ny: int, nx: int, tile_size: tuple[int, int]
) -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]:
    """Returns chunking tuples (length of each chunk in each axis) after tiling.
    I.e., if nx == 2048 and tile_size == 1024, chunks for x axis will be (1024,1024).
    """
    y_tile_size, x_tile_size = tile_size

    y_tiling_chunks = _chunk_by_tile_size(ny, y_tile_size)
    x_tiling_chunks = _chunk_by_tile_size(nx, x_tile_size)

    return ((1,) * nt, (1,) * nc, (1,) * nz, y_tiling_chunks, x_tiling_chunks)


def axis_id_to_slice(axis_id: int, tile_length: int, n_px: int) -> slice:
    """Take the axis_id from a dask block_id and create the corresponding
    tile slice, taking into account edge tiles.
    """
    if (axis_id * tile_length) + tile_length <= n_px:
        return slice(axis_id * tile_length, (axis_id * tile_length) + tile_length)
    else:
        return slice(axis_id * tile_length, n_px)


def slice2width(slc: slice, length: int) -> tuple[int, int]:
    """Convert `slice` object into (start, width)."""
    if slc.stop is not None or slc.start is not None:
        # NOTE: we're ignoring step != 1 here
        start, stop, _ = slc.indices(length)
        return min(start, stop), abs(stop - start)
    return 0, length
