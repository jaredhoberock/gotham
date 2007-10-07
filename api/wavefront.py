import re

def readMesh(filename):
  file = open(filename)
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
        m = pointOnly.match(v)
        try:
          polygon.append((int(m.group(0)) - 1, -1, -1))
        except:
          m = pointUv.match(v)
          try:
            polygon.append((int(m.group(0)) - 1, int(m.group(1)) - 1, -1))
          except:
            m = pointNormal.match(v)
            try:
              polygon.append((int(m.group(0)) - 1, -1, int(m.group(1)) - 1))
            except:
              m = pointUvNormal.match(v)
              try:
                polygon.append((int(m.group(0)) - 1, int(m.group(1)) - 1, int(m.group(2)) - 1))
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

