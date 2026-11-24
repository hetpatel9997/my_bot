#!/usr/bin/env python3
"""
gen_disable_collisions.py  -  make the list of link pairs MoveIt must NOT check for collisions.

Why: MoveIt checks every pair of robot parts for touching. Parts that are bolted together (the
body shell sits on the frame, the tank on the top frame, the arm's base plate in the cover) and
parts joined by one joint (the turret housing standing on the cover, upper arm and forearm at the
elbow) touch BY DESIGN. If MoveIt checked them, every arm pose would be "in collision". So the
SRDF lists those pairs as disabled. This script works the list out from the URDF, so nobody has
to keep it up to date by hand:
  1. Rigid groups: links joined by FIXED joints move together and form one group
     (e.g. base_link + chassis + body shell + tank + LiDAR + phone holder + arm_base, or
     upper_arm + its shoulder servo box).
  2. Disabled pairs: every two links in the same group, and every two links in two groups that are
     joined by ONE moving joint (e.g. the body group and the wheels, the turret and the upper arm).
     Every other pair of links WITH a collision shape is checked, e.g. the upper arm, forearm,
     wrist and nozzle (and their servo boxes) against the body, frame, tank, LiDAR, wheels and
     phone holder, or the turret against the forearm, wrist and nozzle.
  MoveIt cannot check a link that has NO collision shape: such a link is invisible to it, so it
  appears in no pair; the first line of the generated block names all of them. Today (sim URDF,
  2026-11-23) they are:
    - frames without any shape: base_link, base_footprint, laser_frame, phone_mount,
      phone_optical, phone_link;
    - visual-only parts inside the body shell's collision box (the arm would hit body_shell first):
      top_cover, cheek_1, cheek_-1, arm_yaw_servo, imu_link;
    - the caster bracket and socket behind the tank (caster_bracket, caster_socket): out of reach,
      the wrist stays within 0.2 m of the shoulder (x >= -0.19) and the nozzle would have to pass
      through the tank to get behind it;
    - spray_tip, the 6 mm marker ball at the nozzle's end (the nozzle cylinder, which is checked,
      ends at its centre).
  A visual part the arm CAN reach needs a collision shape in the URDF (the phone holder and the
  arm's servo boxes got theirs on 2026-11-23); test_moveit_config.py fails on any new link
  without one until it is added to its allowed list there.
The URDF is the SIM variant (sim_mode true, as arm_aim.launch.py uses in Gazebo).

  python3 test/arm_aim/gen_disable_collisions.py           # print the list
  python3 test/arm_aim/gen_disable_collisions.py --write   # put it into config/moveit/weedbot.srdf
                                                           # between the BEGIN/END generated markers
Needs the workspace sourced (xacro finds my_bot through it). test_moveit_config.py checks that the
SRDF holds exactly this output, so after a URDF change run it with --write.
"""
import argparse
import itertools
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, '..', '..'))
XACRO = os.path.join(PKG, 'description', 'robot.urdf.xacro')
SRDF = os.path.join(PKG, 'config', 'moveit', 'weedbot.srdf')
SIM_MAPPINGS = {'use_ros2_control': 'true', 'sim_mode': 'true',
                'wheel_mu': '1.0', 'sim_camera': 'false'}
BEGIN = '<!-- BEGIN generated -->'
END = '<!-- END generated -->'


def sim_urdf() -> str:
    """The sim robot URDF text (same xacro mappings as launch_sim / arm_aim.launch.py)."""
    import xacro
    return xacro.process_file(XACRO, mappings=SIM_MAPPINGS).toxml()


def rigid_groups(urdf_xml: str):
    """Returns (group, moving): group[link] = name of its rigid group (the group's top link);
    moving = set of frozenset({group_a, group_b}) joined by one moving joint."""
    root = ET.fromstring(urdf_xml)
    links = [link.get('name') for link in root.findall('link')]
    # URDF joints are direct children of <robot>; the <joint> tags inside <ros2_control> are not.
    joints = [(j.get('type'), j.find('parent').get('link'), j.find('child').get('link'))
              for j in root.findall('joint')]
    parent_of = {child: (jtype, parent) for jtype, parent, child in joints}

    def top(link):
        # walk up through fixed joints: the first link reached by a moving joint (or the root)
        # names the group
        while link in parent_of and parent_of[link][0] == 'fixed':
            link = parent_of[link][1]
        return link

    group = {link: top(link) for link in links}
    moving = {frozenset((group[parent], group[child]))
              for jtype, parent, child in joints if jtype != 'fixed'}
    return group, moving


def disable_pairs(urdf_xml: str):
    """Sorted (link1, link2) pairs, link1 < link2, that touch by design (see the docstring)."""
    root = ET.fromstring(urdf_xml)
    solid = sorted(link.get('name') for link in root.findall('link')
                   if link.find('collision') is not None)
    group, moving = rigid_groups(urdf_xml)
    pairs = []
    for a, b in itertools.combinations(solid, 2):      # solid is sorted, so a < b
        if group[a] == group[b] or frozenset((group[a], group[b])) in moving:
            pairs.append((a, b))
    return pairs


def disable_lines(urdf_xml: str):
    """The pairs as SRDF lines (no indentation)."""
    return [f'<disable_collisions link1="{a}" link2="{b}" reason="Adjacent"/>'
            for a, b in disable_pairs(urdf_xml)]


def unchecked_links(urdf_xml: str):
    """Sorted names of the links with no collision shape (MoveIt never checks them)."""
    root = ET.fromstring(urdf_xml)
    return sorted(link.get('name') for link in root.findall('link')
                  if link.find('collision') is None)


def block_lines(urdf_xml: str):
    """Everything between the SRDF markers: one comment naming the never-checked links, then the
    disabled pairs."""
    names = ', '.join(unchecked_links(urdf_xml))
    if '--' in names:        # an XML comment must not contain a double dash
        raise ValueError(f'link name with "--" cannot go into an XML comment: {names}')
    return [f'<!-- never checked (no collision shape): {names} -->'] + disable_lines(urdf_xml)


def read_block(srdf_text: str):
    """The stripped, non-empty lines between the BEGIN and END generated markers."""
    start, end = srdf_text.index(BEGIN) + len(BEGIN), srdf_text.index(END)
    return [line.strip() for line in srdf_text[start:end].splitlines() if line.strip()]


def write_block(srdf_path: str, lines):
    """Replace everything between the markers with `lines` (2-space indent), keep the rest."""
    with open(srdf_path) as f:
        text = f.read()
    start, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    body = ''.join(f'\n  {line}' for line in lines)
    with open(srdf_path, 'w') as f:
        f.write(text[:start] + body + '\n  ' + text[end:])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1].strip())
    ap.add_argument('--write', action='store_true',
                    help=f'replace the generated block in {os.path.relpath(SRDF, PKG)}')
    args = ap.parse_args()
    lines = block_lines(sim_urdf())
    if args.write:
        write_block(SRDF, lines)
        print(f'wrote the never-checked line + {len(lines) - 1} disable_collisions lines '
              f'into {SRDF}')
    else:
        print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
