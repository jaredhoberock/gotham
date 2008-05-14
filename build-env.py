#!/usr/bin/env python
import os
import inspect

def getTools():
  result = []
  if os.name == 'nt':
    result = ['default', 'msvc']
  elif os.name == 'posix':
    #result = ['default', 'intelc']
    # XXX BUG boost_random produces negative numbers in
    #         icc optimized code (-O2 and above)
    # see http://lists.boost.org/Archives/boost/2007/07/124815.php for details
    result = ['default']
  return result

def getReleaseCPPFLAGS():
  result = []
  if os.name == 'nt':
    result = ['/EHsc', '/MD']
  elif os.name == 'posix':
    #result = ['-O3', '-fPIC', '-Wall']
    result = ['-O3', '-fPIC']
  return result

def getDebugCPPFLAGS():
  result = []
  if os.name == 'nt':
    result = ['/EHsc', '/MDd']
  elif os.name == 'posix':
    result = ['-fPIC', '-g', '-Wall']
  return result;

def getEmulationNVCCFLAGS():
  return ['-deviceemu']

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
    #includeMe =  os.path.dirname(os.path.dirname(thisFile))
    includeMe = os.path.join(os.path.dirname(thisFile), 'dependencies')
    result = [includeMe, '/usr/include/python2.5',
              '/usr/include/qt4',
              '/usr/include/qt4/Qt',    '/usr/include/qt4/QtCore',
              '/usr/include/qt4/QtGui', '/usr/include/qt4/QtXml', '/usr/include/qt4/QtOpenGL',
              '/usr/include/OpenEXR',
              '/usr/local/cuda/include']
  return result

def GothamReleaseEnvironment(env):
  env.Append(CPPPATH = getIncludes())
  env.Append(CPPFLAGS = getReleaseCPPFLAGS())

def GothamEmuReleaseEnvironment(env):
  env.Append(CPPPATH = getIncludes())
  env.Append(CPPFLAGS = getReleaseCPPFLAGS())
  env.Append(NVCCFLAGS = getEmulationNVCCFLAGS())

def GothamDebugEnvironment(env):
  env.Append(CPPPATH = getIncludes(),
             CPPFLAGS = getDebugCPPFLAGS())

def GothamEmuDebugEnvironment(env):
  env.Append(CPPPATH = getIncludes(),
             CPPFLAGS = getDebugCPPFLAGS(),
             NVCCFLAGS = getEmulationNVCCFLAGS())

def GothamEnvironment():
  mode = 'release'
  env = Environment(tools = getTools())

  # figure out the absolute path of this file
  thisFile = inspect.getabsfile(GothamEnvironment)
  # include the directory above the one containing this file
  path = os.path.dirname(os.path.dirname(thisFile)) + '/cudascons'

  # enable nvcc
  env.Tool('nvcc', toolpath = [path])

  if ARGUMENTS.get('mode'):
    mode = ARGUMENTS['mode']
  if mode == 'release':
    GothamReleaseEnvironment(env)
  elif mode == 'emurelease':
    GothamEmuReleaseEnvironment(env)
  elif mode == 'debug':
    GothamDebugEnvironment(env)
  elif mode == 'emudebug':
    GothamEmuDebugEnvironment(env)
  else:
    raise ValueError, 'Unknown target mode "%s".' % mode
  return env

