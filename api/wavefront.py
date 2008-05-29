#!/usr/bin/env python
import re
import sys

def readMtllib(filename):
  result = {}
  file = open(filename)
  currentMaterial = ""
  for line in file.readlines():
    words = line.split()
    if len(words) == 0 or words[0] == '#':
      pass
    elif(words[0] == 'newmtl'):
      # add a new material hashed on the name
      # given by the 2nd word
      currentMaterial = words[1]
      result[currentMaterial] = {}
    # XXX find a better way to handle all this
    elif(words[0] == 'map_Kd'):
      result[currentMaterial][words[0]] = words[1]
    elif(words[0] == 'map_Bump'):
      result[currentMaterial][words[0]] = words[1]
    elif(words[0] == 'map_D'):
      result[currentMaterial][words[0]] = words[1]
    elif len(words) == 2:
      result[currentMaterial][words[0]] = float(words[1])
    elif len(words) == 4:
      result[currentMaterial][words[0]] = (float(words[1]), float(words[2]), float(words[3]))
  print 'readMtllib(): parsed %d materials.' % len(result)
  return result

def readMeshes(filename):
  # assume the argument is a filename
  # otherwise, assume it's a file
  try:
    file = open(filename)
  except:
    file = filename

  points = []
  uvs = []
  normals = []
  materials = {}
  materials['gothamDefault'] = {'Kd': (1.0, 1.0, 1.0)}
  primitives = [("gothamDefault", [])]

  pointOnly = re.compile("([0-9]+)")
  pointUv = re.compile("([0-9]+)/([0-9]+)")
  pointNormal = re.compile("([0-9]+)//([0-9]+)")
  pointUvNormal = re.compile("([0-9]+)/([0-9]+)/([0-9]+)")
  for line in file.readlines():
    words = line.split()
    if len(words) == 0 or words[0] == '#':
      pass
    elif(words[0] == 'mtllib'):
      filename = words[1]
      materials.update(readMtllib(filename))
    elif(words[0] == 'usemtl'):
      # create a new primitive
      # append the name of the material and create an empty list of polygons
      primitives.append((words[1], []))
    elif(words[0] == 'v'):
      points.append((float(words[1]), float(words[2]), float(words[3])))
    elif(words[0] == 'vt'):
      uvs.append((float(words[1]), float(words[2])))
    elif(words[0] == 'vn'):
      normals.append((float(words[1]), float(words[2]), float(words[3])))
    elif(words[0] == 'f'):
      n = len(words) - 1
      polygon = []
      if n < 3:
        raise ValueError, "Face is not a polygon!"
      for v in words[1:]:
        m = pointUvNormal.match(v)
        try:
          polygon.append((int(m.group(1)) - 1, int(m.group(2)) - 1, int(m.group(3)) - 1))
        except:
          m = pointNormal.match(v)
          try:
            polygon.append((int(m.group(1)) - 1, -1, int(m.group(2)) - 1))
          except:
            m = pointUv.match(v)
            try:
              polygon.append((int(m.group(1)) - 1, int(m.group(2)) - 1, -1))
            except:
              m = pointOnly.match(v)
              try:
                polygon.append((int(m.group(0)) - 1, -1, -1))
              except:
                raise ValueError, "Unrecognized vertex format: " + v
      # append the new polygon to the current primitive
      primitives[-1][1].append(polygon)
  return (points, uvs, normals, materials, primitives)

def partitionByMaterials(points, uvs, normals, materials, primitives):
  # the idea here is to partition the global lists of points, uvs, normals
  # into a local per-material list of each
  results = []

  # associate with each primitive its unique list of points, uvs, and normals
  for prim in primitives:
    material = prim[0]
    polygons = prim[1]

    # these map from a global index to a local per-material index
    fromGlobalToLocalPoints = {}
    fromGlobalToLocalUVs = {}
    fromGlobalToLocalNormals = {}

    # create a new list of points for each of points, uvs, and normals
    localPoints = []
    localUVs = []
    localNormals = []

    for poly in polygons:
      for vert in poly:
        # for each newly-encountered global index, add a new
        # point to the local list and record a map from the global
        # index to the local index
        if not fromGlobalToLocalPoints.has_key(vert[0]):
          fromGlobalToLocalPoints[vert[0]] = len(localPoints)
          localPoints.append(points[vert[0]])
        if not fromGlobalToLocalUVs.has_key(vert[1]):
          newUV = -1
          if vert[1] != -1:
            newUV = len(localUVs)
            localUVs.append(uvs[vert[1]])
          fromGlobalToLocalUVs[vert[1]] = newUV
        if not fromGlobalToLocalNormals.has_key(vert[2]):
          newNormal = -1;
          if vert[2] != -1:
            newNormal = len(localNormals)
            localNormals.append(normals[vert[2]])
          fromGlobalToLocalNormals[vert[2]] = newNormal
    # now create a new list of polygons with local indices for this primitive
    localPolygons = []
    for poly in polygons:
      localPoly = []
      for vert in poly:
        localPoly.append((fromGlobalToLocalPoints[vert[0]],
                          fromGlobalToLocalUVs[vert[1]],
                          fromGlobalToLocalNormals[vert[2]]))
      localPolygons.append(localPoly)
    results.append((material, localPoints, localUVs, localNormals, localPolygons))
  return (materials, results)

def readMesh(filename):
  # assume the argument is a filename
  # otherwise, assume it's a file
  try:
    file = open(filename)
  except:
    file = filename

  points = []
  uvs = []
  normals = []
  polygons = []

  pointOnly = re.compile("([0-9]+)")
  pointUv = re.compile("([0-9]+)/([0-9]+)")
  pointNormal = re.compile("([0-9]+)//([0-9]+)")
  pointUvNormal = re.compile("([0-9]+)/([0-9]+)/([0-9]+)")
  for line in file.readlines():
    words = line.split()
    if len(words) == 0 or words[0] == '#':
      pass
    elif(words[0] == 'v'):
      points.append((float(words[1]), float(words[2]), float(words[3])))
    elif(words[0] == 'vt'):
      uvs.append((float(words[1]), float(words[2])))
    elif(words[0] == 'vn'):
      normals.append((float(words[1]), float(words[2]), float(words[3])))
    elif(words[0] == 'f'):
      n = len(words) - 1
      polygon = []
      if n < 3:
        raise ValueError, "Face is not a polygon!"
      for v in words[1:]:
        m = pointUvNormal.match(v)
        try:
          polygon.append((int(m.group(1)) - 1, int(m.group(2)) - 1, int(m.group(3)) - 1))
        except:
          m = pointNormal.match(v)
          try:
            polygon.append((int(m.group(1)) - 1, -1, int(m.group(2)) - 1))
          except:
            m = pointUv.match(v)
            try:
              polygon.append((int(m.group(1)) - 1, int(m.group(2)) - 1, -1))
            except:
              m = pointOnly.match(v)
              try:
                polygon.append((int(m.group(0)) - 1, -1, -1))
              except:
                raise ValueError, "Unrecognized vertex format: " + v
      polygons.append(polygon)
  return (points, uvs, normals, polygons)

def triangulate(polygons):
  triangles = []
  for poly in polygons:
    # create a fan
    triangles.extend([(poly[0], v1, v2) for (v1, v2) in zip(poly[1:-1], poly[2:])])
  return triangles

if sys.argv[0] == './wavefront.py':
  if len(sys.argv) < 2:
    print 'usage: wavefront.py input.obj'
  else:
   (points, uvs, normals, materials, primitives) = readMeshes(sys.argv[1])
   primitives = partitionByMaterials(points, uvs, normals, materials, primitives)
   for (material, points, uvs, normals, polygons) in primitives:
     triangles = triangulate(polygons)
     print 'material: ', material
     print 'points: ', points
     print 'uvs: ', uvs
     print 'normals: ', normals
     print 'triangles: ', triangles

