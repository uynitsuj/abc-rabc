"""Scene XML transform helpers for runtime scene variants."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


_FLEXIBLE_TEMPLATE_SCENE = Path(__file__).resolve().parent / "models" / "yam_flexible_chess_scene.xml"
_FLEXIBLE_TEMPLATE_MESH_DIR = _FLEXIBLE_TEMPLATE_SCENE.parent / "assets"
_FLEXIBLE_MESH_NAMES = ("flexible_base", "linear_module", "soft_tips")
_FLEXIBLE_BODY_SPECS = (
    ("left_link_6", {"left_link_left_finger", "left_link_right_finger", "left_flex_gripper"}, "left_flex_gripper"),
    ("right_link_6", {"right_link_left_finger", "right_link_right_finger", "right_flex_gripper"}, "right_flex_gripper"),
)
_FLEXIBLE_EQUALITY_PAIRS = (
    ("left_left_finger", "left_right_finger"),
    ("right_left_finger", "right_right_finger"),
)
_FLEXIBLE_ACTUATORS = ("left_gripper", "right_gripper")
_FLEXIBLE_CONTACT_PAIRS = (
    ("left_flex_gripper", "left_linear_module"),
    ("left_flex_gripper", "left_linear_module_2"),
    ("left_linear_module", "left_linear_module_2"),
    ("right_flex_gripper", "right_linear_module"),
    ("right_flex_gripper", "right_linear_module_2"),
    ("right_linear_module", "right_linear_module_2"),
)
_CLEAN_CAMERA_BODIES = ("overhead_camera", "left_side_camera", "right_side_camera")
_CLEAN_GEOM_NAMES = ("floor", "back_wall", "left_wall", "right_wall")
_CLEAN_GEOM_MESHES = ("base_visual_gate",)


@dataclass(frozen=True)
class SceneXmlTransformOptions:
    clean: bool = False
    mocap: bool = False
    flexible_gripper: bool = False
    debug: bool = False


def _load_xml_root(xml_or_path: str | Path) -> ET.Element:
    if isinstance(xml_or_path, Path):
        return ET.parse(xml_or_path).getroot()
    return ET.fromstring(xml_or_path)


def _xml_to_string(root: ET.Element) -> str:
    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _replace_named_child(parent: ET.Element, tag: str, name: str, new_child: ET.Element) -> None:
    for idx, child in enumerate(list(parent)):
        if child.tag == tag and child.get("name") == name:
            parent.remove(child)
            parent.insert(idx, copy.deepcopy(new_child))
            return
    parent.append(copy.deepcopy(new_child))


def _remove_matching_children(parent: ET.Element, predicate) -> None:
    for child in list(parent):
        if predicate(child):
            parent.remove(child)
            continue
        _remove_matching_children(child, predicate)


def _copy_flexible_mesh_for_scene(root: ET.Element, template_mesh: ET.Element) -> ET.Element:
    mesh = copy.deepcopy(template_mesh)
    file_attr = mesh.get("file")
    if file_attr and not Path(file_attr).is_absolute():
        mesh.set("file", str((_FLEXIBLE_TEMPLATE_MESH_DIR / file_attr).resolve()))
    return mesh


def apply_flexible_gripper_xml(xml: str) -> str:
    root = _load_xml_root(xml)
    template_root = _load_xml_root(_FLEXIBLE_TEMPLATE_SCENE)

    asset = root.find("asset")
    template_asset = template_root.find("asset")
    if asset is None or template_asset is None:
        raise ValueError("Scene XML is missing <asset> section")

    for mesh_name in _FLEXIBLE_MESH_NAMES:
        for child in list(asset):
            if child.tag == "mesh" and child.get("name") == mesh_name:
                asset.remove(child)
        template_mesh = template_asset.find(f"./mesh[@name='{mesh_name}']")
        if template_mesh is None:
            raise ValueError(f"Flexible gripper template is missing mesh '{mesh_name}'")
        asset.append(_copy_flexible_mesh_for_scene(root, template_mesh))

    for parent_name, remove_names, template_name in _FLEXIBLE_BODY_SPECS:
        parent = root.find(f".//body[@name='{parent_name}']")
        template_body = template_root.find(f".//body[@name='{template_name}']")
        if parent is None or template_body is None:
            raise ValueError(f"Unable to locate flexible gripper body '{template_name}'")
        for child in list(parent):
            if child.tag == "body" and child.get("name") in remove_names:
                parent.remove(child)
        parent.append(copy.deepcopy(template_body))

    equality = root.find("equality")
    template_equality = template_root.find("equality")
    if equality is None or template_equality is None:
        raise ValueError("Scene XML is missing <equality> section")
    for child in list(equality):
        if child.tag != "joint":
            continue
        pair = (child.get("joint1"), child.get("joint2"))
        if pair in _FLEXIBLE_EQUALITY_PAIRS or pair[::-1] in _FLEXIBLE_EQUALITY_PAIRS:
            equality.remove(child)
    for joint1, joint2 in _FLEXIBLE_EQUALITY_PAIRS:
        template_joint = template_equality.find(f"./joint[@joint1='{joint1}'][@joint2='{joint2}']")
        if template_joint is None:
            raise ValueError(f"Flexible gripper template is missing equality joint {joint1}/{joint2}")
        equality.append(copy.deepcopy(template_joint))

    actuator = root.find("actuator")
    template_actuator = template_root.find("actuator")
    if actuator is None or template_actuator is None:
        raise ValueError("Scene XML is missing <actuator> section")
    for actuator_name in _FLEXIBLE_ACTUATORS:
        template_position = template_actuator.find(f"./position[@name='{actuator_name}']")
        if template_position is None:
            raise ValueError(f"Flexible gripper template is missing actuator '{actuator_name}'")
        _replace_named_child(actuator, "position", actuator_name, template_position)

    template_contact = template_root.find("contact")
    contact = root.find("contact")
    if template_contact is not None:
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for child in list(contact):
            if child.tag != "exclude":
                continue
            pair = (child.get("body1"), child.get("body2"))
            if pair in _FLEXIBLE_CONTACT_PAIRS or pair[::-1] in _FLEXIBLE_CONTACT_PAIRS:
                contact.remove(child)
        seen_pairs: set[tuple[str | None, str | None]] = set()
        for exclude in template_contact.findall("./exclude"):
            pair = (exclude.get("body1"), exclude.get("body2"))
            if pair not in _FLEXIBLE_CONTACT_PAIRS or pair in seen_pairs:
                continue
            contact.append(copy.deepcopy(exclude))
            seen_pairs.add(pair)

    return _xml_to_string(root)


def apply_clean_xml(xml: str) -> str:
    root = _load_xml_root(xml)

    _remove_matching_children(
        root,
        lambda child: child.tag == "geom"
        and (
            child.get("name") in _CLEAN_GEOM_NAMES
            or child.get("mesh") in _CLEAN_GEOM_MESHES
        ),
    )
    _remove_matching_children(
        root,
        lambda child: child.tag == "body"
        and child.get("name") in (*_CLEAN_CAMERA_BODIES, "top_camera_d405"),
    )
    return _xml_to_string(root)


def apply_mocap_xml(xml: str, *, debug: bool) -> str:
    root = _load_xml_root(xml)
    _remove_matching_children(
        root,
        lambda child: child.tag == "body"
        and child.get("name") in {"left_mocap", "right_mocap"},
    )
    equality = root.find("equality")
    if equality is not None:
        for child in list(equality):
            if child.tag == "weld" and child.get("name") in {"left_mocap_weld", "right_mocap_weld"}:
                equality.remove(child)
    xml = _xml_to_string(root)

    left_mocap_geom = ""
    right_mocap_geom = ""
    if debug:
        left_mocap_geom = '\n      <geom type="box" size="0.02 0.02 0.02" contype="0" conaffinity="0" rgba="0.2 0.9 0.2 0.3" group="2"/>'
        right_mocap_geom = '\n      <geom type="box" size="0.02 0.02 0.02" contype="0" conaffinity="0" rgba="0.9 0.2 0.2 0.3" group="2"/>'
    mocap_xml = f"""
    <!-- Mocap targets for VR controller arm control -->
    <body mocap="true" name="left_mocap" pos="0.6295 0.3100 1.1426" quat="0.7071 0 0.7071 0">
      <site name="left_mocap_site" size="0.01" type="sphere" rgba="0.2 0.9 0.2 0.5"/>{left_mocap_geom}
    </body>
    <body mocap="true" name="right_mocap" pos="0.6295 -0.3100 1.1426" quat="0.7071 0 0.7071 0">
      <site name="right_mocap_site" size="0.01" type="sphere" rgba="0.9 0.2 0.2 0.5"/>{right_mocap_geom}
    </body>
"""
    xml = xml.replace("</worldbody>", mocap_xml + "  </worldbody>")
    weld_xml = (
        '    <weld name="left_mocap_weld" site1="left_mocap_site" site2="left_grasp_site"/>\n'
        '    <weld name="right_mocap_weld" site1="right_mocap_site" site2="right_grasp_site"/>\n'
    )
    if "<equality>" in xml:
        xml = xml.replace("</equality>", weld_xml + "  </equality>")
    else:
        xml = xml.replace("</worldbody>", "</worldbody>\n  <equality>\n" + weld_xml + "  </equality>")
    return xml


def transform_scene_xml(
    xml: str,
    *,
    options: SceneXmlTransformOptions,
) -> tuple[str, tuple[str, ...]]:
    edits: list[str] = []
    if options.flexible_gripper:
        xml = apply_flexible_gripper_xml(xml)
        edits.append("flexible_gripper")
    if options.clean:
        xml = apply_clean_xml(xml)
        edits.append("clean")
    if options.mocap:
        xml = apply_mocap_xml(xml, debug=options.debug)
        edits.append("mocap")
    return xml, tuple(edits)


def build_scene_xml(
    scene_path: Path,
    *,
    options: SceneXmlTransformOptions,
) -> tuple[str, tuple[str, ...]]:
    with open(scene_path) as f:
        xml = f.read()

    return transform_scene_xml(xml, options=options)


__all__ = [
    "SceneXmlTransformOptions",
    "apply_clean_xml",
    "apply_flexible_gripper_xml",
    "apply_mocap_xml",
    "build_scene_xml",
    "transform_scene_xml",
]
