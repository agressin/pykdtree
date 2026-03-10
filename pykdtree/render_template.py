#!/usr/bin/env python

from mako.template import Template

mytemplate = Template(filename='_kdtree_core.c.mako')
with open('_kdtree_core.c', 'w') as fp:
    fp.write(mytemplate.render())

spatial_template = Template(filename='_spatial_ops.c.mako')
with open('_spatial_ops.c', 'w') as fp:
    fp.write(spatial_template.render())
