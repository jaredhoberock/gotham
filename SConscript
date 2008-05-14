import os

Import('env')

# add dependencies
# with msvc, boost figures out which libraries to link
if os.name == 'posix':
  env.Append(LIBS = ['boost_python'])
  env.Append(LIBS = ['boost_thread'])

# openexr libraries
env.Append(LIBS = ['Iex'])
env.Append(LIBS = ['IlmImf'])
env.Append(LIBS = ['Imath'])
env.Append(LIBS = ['Half'])
env.Append(LIBS = ['z'])

# qt libraries
env.Append(LIBS = ['QtCore'])
env.Append(LIBS = ['QtXml'])
env.Append(LIBS = ['QtGui'])
env.Append(LIBS = ['QtOpenGL'])

# glut libraries
env.Append(LIBS = ['glut'])

# gl libraries
env.Append(LIBS = ['GL'])
env.Append(LIBS = ['GLU'])
env.Append(LIBS = ['GLEW'])
env.Append(LIBS = ['QGLViewer'])

# aggregate shared objects from each child directory
subdirectories = ['geometry',
                  'importance',
                  'mutators',
                  'path',
                  'primitives',
                  'rasterizables',
                  'records',
                  'renderers',
                  'shading',
                  'surfaces',
                  'api']
sharedObjects = []
for dir in subdirectories:
  sharedObjects += SConscript('%s/SConscript' % dir, exports={'env':env})

# create a shared library
gothamLib = env.SharedLibrary('gotham', sharedObjects)

# install the distribution
try:
  dir = os.environ["GOTHAMHOME"] + '/lib'
  Default(env.Install(dir, source = gothamLib))
  Default(env.Install(dir, source = 'api/api.py'))
  Default(env.Install(dir, source = 'api/objtogoth.py'))
  Default(env.Install(dir, source = 'api/wavefront.py'))
  dir = os.environ["GOTHAMHOME"] + '/bin'
  Default(env.Install(dir, source = 'api/pbslc'))
  Default(env.Install(dir, source = 'api/gotham'))
except:
  print "Warning: $GOTHAMHOME not defined! Gotham could not be installed."

# build imgstat
env.Append(LIBS = ['python2.5'])
env.Program('imgstat', 'api/imgstat.cpp')
dir = os.environ["GOTHAMHOME"] + '/bin'
Default(env.Install(dir, source = 'imgstat'))

# build imgclean
env.Append(LIBS = ['python2.5'])
env.Program('imgclean', 'api/imgclean.cpp')
dir = os.environ["GOTHAMHOME"] + '/bin'
Default(env.Install(dir, source = 'imgclean'))

