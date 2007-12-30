#!/usr/bin/env python
import sys
import wavefront

def splitTriangles(points, triangles):
  newPoints = []
  newTriangles = []
  index = 0
  for (v0,v1,v2) in triangles:
    newPoints.append(points[v0[0]])
    newPoints.append(points[v1[0]])
    newPoints.append(points[v2[0]])
    newV0 = (index,     -1, -1)
    newV1 = (index + 1, -1, -1)
    newV2 = (index + 2, -1, -1)
    newTriangles.append([newV0, newV1, newV2])
    index += 3
  return (newPoints, newTriangles)

def splitTrianglesWithParametrics(points, uvs, triangles):
  newPoints = []
  newUvs = []
  newTriangles = []
  index = 0
  for (v0,v1,v2) in triangles:
    newPoints.append(points[v0[0]])
    newPoints.append(points[v1[0]])
    newPoints.append(points[v2[0]])
    newUvs.append(uvs[v0[1]])
    newUvs.append(uvs[v1[1]])
    newUvs.append(uvs[v2[1]])
    newV0 = (index,     index,     -1)
    newV1 = (index + 1, index + 1, -1)
    newV2 = (index + 2, index + 2, -1)
    newTriangles.append([newV0, newV1, newV2])
    index += 3
  return (newPoints, newUvs, newTriangles)

def objtogoth(filename):
  (points, uvs, normals, polygons) = wavefront.readMesh(filename)
  triangles = wavefront.triangulate(polygons)
  if uvs != []:
    (points, uvs, triangles) = splitTrianglesWithParametrics(points, uvs, triangles)
  else:
    (points, triangles) = splitTriangles(points, triangles)

  vertices = []
  indices = []
  parms = []
  # flatten the points list
  for p in points:
    vertices.extend(p)
  # flatten the uv list
  for uv in uvs:
    parms.extend(uv)
  # flatten the triangle list
  for tri in triangles:
    indices.append(tri[0][0])
    indices.append(tri[1][0])
    indices.append(tri[2][0])
  return (vertices, parms, indices)

# did we call this as a program?
if sys.argv[0] == './objtogoth.py':
  if len(sys.argv) < 2:
    print 'usage: objtogoth input.obj'
  else:
    (vertices, uvs, triangles) = objtogoth(sys.argv[1])
    # print a mesh call
    print 'g.mesh(',
    print vertices,
    print ',',
    print uvs,
    print ',',
    print triangles,
    print ')'
  
