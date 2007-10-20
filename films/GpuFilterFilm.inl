/*! \file GpuFilterFilm.inl
 *  \author Jared Hoberock
 *  \brief Inline file for GpuFilterFilm.h.
 */

#include "GpuFilterFilm.h"
#include <stratifiedsequence/StratifiedSequence.h>

template<typename ParentFilmType>
  void GpuFilterFilm<ParentFilmType>
    ::init(void)
{
  Parent::init();
  reloadShaders();
} // end GpuFilterFilm::init()

// XXX DESIGN this pollutes the global namespace
static float filter(const float x, const float a, const float w)
{
  return expf(-a * x*x) - expf(-a * w*w);
} // end filter()

template<typename ParentFilmType>
  void GpuFilterFilm<ParentFilmType>
    ::reloadShaders(void)
{
  std::string source = "\n\
    uniform ivec2 imageDim;\n\
    void main(void)\n\
    {\n\
      gl_FrontColor = gl_Color;\n\
      gl_FrontColor.a = 1.0;\n\
      gl_Position = vec4(gl_Vertex.xy, 0.0, 1.0);\n\
      // output the sample position's pixel coordinates on texcoord 0\n\
      gl_TexCoord[0].xy = vec2(imageDim) * gl_Vertex.xy;\n\
    }";
  mDepositVertexShader.create(GL_VERTEX_SHADER, source.c_str());

  source = "\n\
    #version 120\n\
    #extension GL_EXT_geometry_shader4 : enable\n\
    uniform ivec2 imageDim;\n\
    uniform vec2 invImageDim;\n\
    uniform float filterWidth;\n\
    void main(void)\n\
    {\n\
      gl_FrontColor = gl_FrontColorIn[0];\n\
      gl_Position.zw = vec2(0.0,1.0);\n\
      // output the sample position on texcoord 0\n\
      gl_TexCoord[0].xy = vec2(imageDim) * gl_PositionIn[0].xy;\n\
      // transform sample in [0,1)^2 to [0,width) x [0,height)\n\
      vec2 sample = gl_PositionIn[0].xy;\n\
      sample *= vec2(imageDim);\n\
      // output the quad surrounding this sample:\n\
      // lower left:\n\
      gl_Position.xy = sample + vec2(-filterWidth, -filterWidth);\n\
      gl_Position.xy = vec2(floor(gl_Position.x), floor(gl_Position.y));\n\
      // transform back to [0,1)^2\n\
      gl_Position.xy *= invImageDim;\n\
      // transform to [-1,-1)^2\n\
      gl_Position.xy *= 2.0;\n\
      gl_Position.xy -= vec2(1.0,1.0);\n\
      EmitVertex();\n\
      // lower right:\n\
      gl_Position.xy = sample + vec2(filterWidth, -filterWidth);\n\
      gl_Position.xy = vec2(ceil(gl_Position.x), floor(gl_Position.y));\n\
      // transform back to [0,1)^2\n\
      gl_Position.xy *= invImageDim;\n\
      // transform to [-1,-1)^2\n\
      gl_Position.xy *= 2.0;\n\
      gl_Position.xy -= vec2(1.0,1.0);\n\
      EmitVertex();\n\
      // upper left:\n\
      gl_Position.xy = sample + vec2(-filterWidth, filterWidth);\n\
      gl_Position.xy = vec2(floor(gl_Position.x), ceil(gl_Position.y));\n\
      // transform back to [0,1)^2\n\
      gl_Position.xy *= invImageDim;\n\
      // transform to [-1,-1)^2\n\
      gl_Position.xy *= 2.0;\n\
      gl_Position.xy -= vec2(1.0,1.0);\n\
      EmitVertex();\n\
      // upper right:\n\
      gl_Position.xy = sample + vec2(filterWidth, filterWidth);\n\
      gl_Position.xy = ceil(gl_Position.xy);\n\
      // transform back to [0,1)^2\n\
      gl_Position.xy *= invImageDim;\n\
      // transform to [-1,-1)^2\n\
      gl_Position.xy *= 2.0;\n\
      gl_Position.xy -= vec2(1.0,1.0);\n\
      EmitVertex();\n\
      EndPrimitive();\n\
    }";
  mDepositGeometryShader.create(GL_GEOMETRY_SHADER_EXT, source.c_str());

  source = "\n\
    #version 120\n\
    uniform float a;\n\
    uniform float area;\n\
    uniform float filterWidth;\n\
    float filter(const float x)\n\
    {\n\
      return exp(-a * x*x) - exp(-a * filterWidth * filterWidth);\n\
    } // end filter()\n\
    void main(void)\n\
    {\n\
      // get distance in pixels from sample location\n\
      vec2 x = gl_TexCoord[0].xy - gl_FragCoord.xy;\n\
      // evaluate filter\n\
      gl_FragColor = gl_Color;\n\
      gl_FragColor *= filter(x.x) * filter(x.y);\n\
      // weight the pixel by 1 over the number of pixels affected by the sample times\n\
      // the integral of the filter\n\
      gl_FragColor /= (4.0 * filterWidth * filterWidth);\n\
      gl_FragColor /= area;\n\
      gl_FragColor.a = 1.0;\n\
      // clamp to zero\n\
      gl_FragColor.rgb = clamp(gl_FragColor.rgb, vec3(0,0,0), gl_FragColor.rgb);\n\
    }";
  mDepositFragmentShader.create(GL_FRAGMENT_SHADER, source.c_str());

  mDepositProgram.create();
  mDepositProgram.setParameteri(GL_GEOMETRY_VERTICES_OUT_EXT, 4);
  mDepositProgram.setParameteri(GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
  mDepositProgram.setParameteri(GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);
  if(!mDepositProgram.link(mDepositVertexShader, mDepositGeometryShader, mDepositFragmentShader))
  {
    std::cerr << "GpuFilterFilm::reloadShaders(): Problem linking mDepositProgram." << std::endl;
    std::cerr << mDepositProgram << std::endl;
  } // end if

  float w = 1.5f;
  float a = 1.5f;

  // integrate filter's "area"
  float area = 0;
  StratifiedSequence seq(0, 1, 0, 1, 100, 100);
  float x,y;
  while(seq(x,y))
  {
    // map [0,1) to [-w,w)
    x *= 2.0f * w;
    x -= w;
    y *= 2.0f * w;
    y -= w;

    area += filter(x, a, w) * filter(y, a, w);
  } // end for i
  area /= (100*100);

  mDepositProgram.bind();
  mDepositProgram.setUniform1f("filterWidth", w);
  mDepositProgram.setUniform1f("a", a);
  mDepositProgram.setUniform1f("area", area);
  mDepositProgram.unbind();
} // end GpuFilterFilm::reloadShaders()

template<typename ParentFilmType>
  void GpuFilterFilm<ParentFilmType>
    ::renderPendingDeposits(void)
{
  // setup the framebuffer
  glPushAttrib(GL_VIEWPORT_BIT | GL_TRANSFORM_BIT | GL_COLOR_BUFFER_BIT | GL_POINT_BIT | GL_CURRENT_BIT | GL_LIGHTING_BIT);

  // identity xfrm
  // XXX PERF we don't need this with user shaders
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // set up the projection [0,1)^2
  gluOrtho2D(0, 1, 0, 1);

  glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
  glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

  // set up the viewport
  glViewport(0,0,
             Parent::mTexture.getWidth(),
             Parent::mTexture.getHeight());

  // bind the framebuffer
  Parent::mFramebuffer.bind();
  printGLError(__FILE__, __LINE__);

  // XXX PERF attach this once in the init or something
  Parent::mFramebuffer.attachTexture(Parent::mTexture.getTarget(),
                                     GL_COLOR_ATTACHMENT0_EXT,
                                     Parent::mTexture);

  glEnable(GL_BLEND);
  glDisable(GL_LIGHTING);
  glBlendFunc(GL_ONE, GL_ONE);
  glPointSize(1.0f);

  checkFramebufferStatus(__FILE__, __LINE__);

  mDepositProgram.bind();
  mDepositProgram.setUniform2i("imageDim", Parent::getWidth(), Parent::getHeight());
  mDepositProgram.setUniform2f("invImageDim", 1.0f / Parent::getWidth(), 1.0f / Parent::getHeight());

  // XXX PERF replace this with a glDrawElements call
  glBegin(GL_POINTS);
  boost::mutex::scoped_lock lock(Parent::mMutex);
  for(size_t i = 0;
      i != Parent::mDepositBuffer.size();
      ++i)
  {
    glColor3fv(Parent::mDepositBuffer[i].second);
    glVertex2fv(Parent::mDepositBuffer[i].first);
  } // end for i
  lock.unlock();
  glEnd();

  mDepositProgram.unbind();

  // XXX PERF no need to attach/detach each time
  Parent::mFramebuffer.detach(GL_COLOR_ATTACHMENT0_EXT);
  Parent::mFramebuffer.unbind();

  // restore matrix state
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glPopAttrib();

  // clear pending deposits
  lock.lock();
  Parent::mDepositBuffer.clear();
  lock.unlock();
} // end GpuFilterFilm::renderPendingDeposits()

