#!python
from gotham import *
import os

class Gotham2(Gotham):
  # standard shaderpaths
  shaderpaths = ["/home/jared/dev/src/gotham/shading"]

  def material(self, name):
    # add .so to the end if it doesn't exist
    filename = name
    ext = ''
    if os.name == 'posix':
      ext = '.so'
    elif os.name == 'nt':
      ext = '.dll'
    else:
      raise ValueError, "Unknown operating system!" 
    if filename[-3:] != ext:
      filename += ext
    # search each directory in the shaderpath
    # use the first one found
    foundshader = ''
    for path in self.shaderpaths:
      trypath = os.path.join(path,filename)
      if os.path.exists(trypath):
        foundshader = trypath
        break
    result = foundshader != '' and Gotham.material(self, foundshader)
    if not result:
      print "Unable to find material '%s'." % name
    return result

