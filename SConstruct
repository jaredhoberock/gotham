exec open('build-env.py')

env = GothamEnvironment()
SConscript(['geometry/SConscript',
            'importance/SConscript',
            'mutators/SConscript',
            'path/SConscript',
            'primitives/SConscript',
            'rasterizables/SConscript',
            'records/SConscript',
            'renderers/SConscript',
            'shading/SConscript',
            'surfaces/SConscript',
            'api/SConscript'], exports={'env':env})

# XXX why doesn't this do anything???
SConscript(['shaders/SConstruct'])

