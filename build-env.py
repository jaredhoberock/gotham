#!/usr/bin/env python
import os
import inspect

def getTools():
  result = []
  if os.name == 'nt':
    result = ['default', 'msvc']
  elif os.name == 'posix':
    #result = ['default', 'intelc']
    # XXX BUG after upgrading Ubuntu to Gutsy
    #         gotham segfaults when compiled with icc
    result = ['default']
  return result

def getReleaseCPPFLAGS():
  result = []
  if os.name == 'nt':
    result = ['/EHsc', '/MD']
  elif os.name == 'posix':
    result = ['-O3', '-fPIC', '-Wall']
  return result

def getDebugCPPFLAGS():
  result = []
  if os.name == 'nt':
    result = ['/EHsc', '/MDd']
  elif os.name == 'posix':
    result = ['-fPIC', '-g', '-Wall']
  return result;

def getIncludes():
  result = []
  if os.name == 'nt':
    result = ['c:/dev/src/', 'c:/dev/include', 'c:/Python25/include',
              'c:/dev/include/Qt', 'c:/dev/include/QtCore',
              'c:/dev/include/QtGui', 'c:/dev/include/QtXml', 'c:/dev/include/QtOpenGL',
              'c:/dev/include/OpenEXR']
  elif os.name == 'posix':
    # figure out the absolute path of this file
    thisFile = inspect.getabsfile(getIncludes)
    # include the directory above the one containing this file
    includeMe =  os.path.dirname(os.path.dirname(thisFile))
    result = [includeMe, '/usr/include/python2.5',
              '/usr/include/qt4',
              '/usr/include/qt4/Qt',    '/usr/include/qt4/QtCore',
              '/usr/include/qt4/QtGui', '/usr/include/qt4/QtXml', '/usr/include/qt4/QtOpenGL',
              '/usr/include/OpenEXR']
  return result

def GothamReleaseEnvironment():
  includes = getIncludes()
  return Environment(CPPPATH = includes,
                     tools = getTools(),
                     CPPFLAGS = getReleaseCPPFLAGS())

def GothamDebugEnvironment():
  includes = getIncludes()
  return Environment(CPPPATH = includes,
                     tools = getTools(),
                     CPPFLAGS = getDebugCPPFLAGS())

def GothamEnvironment():
  mode = 'release'
  if ARGUMENTS.get('mode'):
    mode = ARGUMENTS['mode']
  if mode == 'release':
    return GothamReleaseEnvironment()
  elif mode == 'debug':
    return GothamDebugEnvironment()
  else:
    raise ValueError, 'Unknown target mode "%s".' % mode

