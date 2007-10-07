#!/usr/bin/env python
import sys
import wavefront

def objtogoth(filename):
  (points, uvs, normals, polygons) = wavefront.readMesh(filename)
  triangles = wavefront.triangulate(polygons)
  vertices = []
  indices = []
  # flatten the points list
  for p in points:
    vertices.extend(p)
  # flatten the triangle list
  for tri in triangles:
    indices.append(tri[0][0])
    indices.append(tri[1][0])
    indices.append(tri[2][0])
  return (vertices, indices)

# did we call this as a program?
if sys.argv[0] == './objtogoth.py':
  if len(sys.argv) < 2:
    print 'usage: objtogoth input.obj'
  else:
    (vertices, triangles) = objtogoth(sys.argv[1])
    # print a mesh call
    print 'g.mesh(',
    print vertices,
    print ',',
    print triangles,
    print ')'
  
