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
     (e.g. base_link + chassis + body shell + tank + LiDAR + arm_base).
  2. Disabled pairs: every two links in the same group, and every two links in two groups that are
     joined by ONE moving joint (e.g. the body group and the wheels, the turret and the upper arm).
     Everything else is still checked, e.g. the nozzle against the body or the forearm against
     the tank.
  Only links that have collision geometry are listed: a link without any (frames like laser_frame,
  visual-only servo boxes) can never collide, so a pair with it would only make the list longer
  (MoveIt Setup Assistant leaves them out too).
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
    lines = disable_lines(sim_urdf())
    if args.write:
        write_block(SRDF, lines)
        print(f'wrote {len(lines)} disable_collisions lines into {SRDF}')
    else:
        print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
