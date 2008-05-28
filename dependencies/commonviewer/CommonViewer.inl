/*! \file CommonViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CommonViewer.h.
 */

#include "CommonViewer.h"
#include <printglerror/printGLError.h>

#ifdef QT_VERSION
// #include qt stuff for file dialog if we're using Qt
#include <qfiledialog.h>
#include <qstring.h>
#endif // QT_VERSION

template<typename Parent, typename KeyEvent, typename StringType, typename VectorType>
  void CommonViewer<Parent,KeyEvent,StringType,VectorType>
    ::init(void)
{
  // init glew
  glewInit();

  // these extensions are vital
  std::vector<std::string> extensions;
  extensions.push_back("GL_ARB_fragment_shader");
  extensions.push_back("GL_ARB_texture_rectangle");

  // check for the GL extensions we plan to use
  for(std::vector<std::string>::const_iterator ext = extensions.begin();
      ext != extensions.end();
      ++ext)
  {
    if(!glewIsSupported(ext->c_str()))
    {
      std::cerr << "CommonViewer::init(): Error: " << *ext << " is not supported on this system." << std::endl;
      throw;
    } // end if
  } // end for ext

  // call the parent
  Parent::init();

  // compile the shaders
  reloadShaders();
} // end CommonViewer::init()

template<typename Parent, typename KeyEvent, typename StringType, typename VectorType>
  void CommonViewer<Parent,KeyEvent,StringType,VectorType>
    ::drawTexture(const glpp::Texture &t,
                  const Program &p) const
{
  glPushAttrib(GL_DEPTH_BUFFER_BIT |
               GL_LIGHTING_BIT | GL_TRANSFORM_BIT);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  t.bind();

  p.bind();
  printGLError(__FILE__, __LINE__);

  glBegin(GL_QUADS);
  glTexCoord2f(0,0);
  glVertex2f(-1,-1);
  glTexCoord2f(t.getMaxS(), 0);
  glVertex2f(1,-1);
  glTexCoord2f(t.getMaxS(), t.getMaxT());
  glVertex2f(1,1);
  glTexCoord2f(0, t.getMaxT());
  glVertex2f(-1,1);
  glEnd();

  p.unbind();

  t.unbind();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glPopAttrib();
} // end CommonViewer::drawTexture()

template<typename Parent, typename KeyEvent, typename StringType, typename VectorType>
  void CommonViewer<Parent,KeyEvent,StringType,VectorType>
    ::drawTexture(const glpp::Texture &t) const
{
#if 0
  const Program &p = (t.getTarget() == GL_TEXTURE_2D_ARRAY_EXT) ? mTexture2DArrayProgram : mTexture2DRectProgram;
#else
  const Program &p = mTexture2DRectProgram;
  std::cerr << "CommonViewer::drawTexture(): glew doesn't support GL_TEXTURE_2D_ARRAY_EXT." << std::endl;
#endif // 0
  drawTexture(t,p);
} // end CommonViewer::drawTexture()

template<typename Parent, typename KeyEvent, typename StringType, typename VectorType>
  void CommonViewer<Parent,KeyEvent,StringType,VectorType>
    ::keyPressEvent(KeyEvent *e)
{
  switch(e->key())
  {
    case 'R':
    {
      reloadShaders();
      Parent::updateGL();
      break;
    } // end case Qt::Key_R

    default:
    {
      Parent::keyPressEvent(e);
      break;
    } // end default
  } // end switch
} // end CommonViewer::keyPressEvent()

template<typename Parent, typename KeyEvent, typename StringType, typename VectorType>
  void CommonViewer<Parent,KeyEvent,StringType,VectorType>
    ::reloadShaders(void)
{
  std::string source = "\n\
    #extension GL_ARB_texture_rectangle : enable\n\
    uniform sampler2DRect texture;\n\
    uniform float scale;\n\
    void main(void)\n\
    {\n\
      gl_FragColor = scale * gl_Color * texture2DRect(texture, gl_TexCoord[0].xy);\n\
      if(gl_FragColor.r != gl_FragColor.r ||\n\
         gl_FragColor.g != gl_FragColor.g ||\n\
         gl_FragColor.b != gl_FragColor.b)\n\
      {\n\
        gl_FragColor.rgb = vec3(1,0,0);\n\
      }\n\
    }";
  mTexture2DRectShader.create(GL_FRAGMENT_SHADER, source.c_str());

  if(!mTexture2DRectProgram.create(0,0,mTexture2DRectShader))
  {
    std::cerr << "CommonViewer::reloadShaders(): Problem creating mTexture2DRectProgram." << std::endl;
    std::cerr << mTexture2DRectProgram << std::endl;
  } // end if

  // set texunit # & default scale
  mTexture2DRectProgram.bind();
  mTexture2DRectProgram.setUniform1i("texture", 0);
  mTexture2DRectProgram.setUniform1f("scale", 1.0f);
  mTexture2DRectProgram.unbind();

  source = "\n\
    #extension GL_ARB_texture_rectangle : enable\n\
    uniform sampler2DRect texture;\n\
    uniform float scale;\n\
    uniform float gamma;\n\
    void main(void)\n\
    {\n\
      gl_FragColor = texture2DRect(texture, gl_TexCoord[0].xy);\n\
      gl_FragColor.rgb = pow(scale * gl_Color * gl_FragColor.rgb, 1.0f / gamma);\n\
      if(gl_FragColor.r != gl_FragColor.r ||\n\
         gl_FragColor.g != gl_FragColor.g ||\n\
         gl_FragColor.b != gl_FragColor.b)\n\
      {\n\
        gl_FragColor.rgb = vec3(1,0,0);\n\
      }\n\
    }";
  mTexture2DRectGammaShader.create(GL_FRAGMENT_SHADER, source.c_str());
  if(!mTexture2DRectGammaProgram.create(0,0,mTexture2DRectGammaShader))
  {
    std::cerr << "CommonViewer::reloadShaders(): Problem creating mTexture2DRectGammaProgram." << std::endl;
    std::cerr << mTexture2DRectGammaProgram << std::endl;
  } // end if

  // set texunit # & default state
  mTexture2DRectGammaProgram.bind();
  mTexture2DRectGammaProgram.setUniform1i("texture", 0);
  mTexture2DRectGammaProgram.setUniform1f("scale", 1.0);
  mTexture2DRectGammaProgram.setUniform1f("gamma", 1.0);
  mTexture2DRectGammaProgram.unbind();

  // we can get by without this program if
  // the extensions aren't supported
  if(glewIsSupported("GL_EXT_texture_array")
     && glewIsSupported("GL_EXT_gpu_shader4"))
  {
    source = "\n\
      #extension GL_EXT_texture_array : enable\n\
      #extension GL_EXT_gpu_shader4 : enable\n\
      uniform sampler2DArray texture;\n\
      void main(void)\n\
      {\n\
        gl_FragColor = texture2DArrayLod(texture, vec3(gl_TexCoord[0].xy,1), 0.0f);\n\
      }";
    mTexture2DArrayShader.create(GL_FRAGMENT_SHADER, source.c_str());

    if(!mTexture2DArrayProgram.create(0,0,mTexture2DArrayShader))
    {
      std::cerr << "CommonViewer::reloadShaders(): Problem creating mTexture2DArrayProgram." << std::endl;
      std::cerr << mTexture2DArrayProgram << std::endl;
    } // end if

    // set texunit #
    mTexture2DArrayProgram.bind();
    mTexture2DArrayProgram.setUniform1i("texture", 0);
    mTexture2DArrayProgram.unbind();
  } // end if

  source = "\n\
    #extension GL_ARB_texture_rectangle : enable\n\
    uniform sampler2DRect texture;\n\
    void main(void)\n\
    {\n\
      gl_FragColor = texture2DRect(texture, gl_TexCoord[0].xy);\n\
      gl_FragColor /= gl_FragColor.a;\n\
    }";
  mTexture2DRectNormalizeShader.create(GL_FRAGMENT_SHADER, source.c_str());

  if(!mTexture2DRectNormalizeProgram.create(0,0,mTexture2DRectNormalizeShader))
  {
    std::cerr << "CommonViewer::reloadShaders(): Problem creating mTexture2DRectNormalizeProgram." << std::endl;
    std::cerr << mTexture2DRectNormalizeProgram << std::endl;
  } // end if

  // set texunit #
  mTexture2DRectNormalizeProgram.bind();
  mTexture2DRectNormalizeProgram.setUniform1i("texture", 0);
  mTexture2DRectNormalizeProgram.unbind();

  source = "\n\
    #extension GL_ARB_texture_rectangle : enable\n\
    uniform sampler2DRect texture;\n\
    uniform float a;\n\
    uniform float Lwa;\n\
    uniform float Lwhite;\n\
    // Reinhard et al 2002, equation 2:\n\
    float L(float Lw)\n\
    {\n\
      return a * Lw / Lwa;\n\
    }\n\
    void main(void)\n\
    {\n\
      gl_FragColor = gl_Color * texture2DRect(texture, gl_TexCoord[0].xy);\n\
      // compute the world luminance of the pixel\n\
      float Lw = dot(gl_FragColor, vec4(0.299, 0.587, 0.114, 0));\n\
      float Lpixel = L(Lw);\n\
      float Lmax = L(Lwhite);\n\
      float Lmax2 = Lmax * Lmax;\n\
      float Ld = Lpixel * (1.0 + Lpixel / Lmax2);\n\
      Ld /= (1.0 + Lpixel);\n\
      if(Lw > 0) gl_FragColor.rgb /= Lw;\n\
      gl_FragColor.rgb *= Ld;\n\
      gl_FragColor.a = 1.0;\n\
      if(gl_FragColor.r != gl_FragColor.r ||\n\
         gl_FragColor.g != gl_FragColor.g ||\n\
         gl_FragColor.b != gl_FragColor.b)\n\
      {\n\
        gl_FragColor.rgb = vec3(1,0,0);\n\
      }\n\
    }";
  mTexture2DRectTonemapShader.create(GL_FRAGMENT_SHADER, source.c_str());
  if(!mTexture2DRectTonemapProgram.create(0,0,mTexture2DRectTonemapShader))
  {
    std::cerr << "CommonViewer::reloadShaders(): Problem creating mTexture2DRectTonemapProgram." << std::endl;
    std::cerr << mTexture2DRectTonemapProgram << std::endl;
  } // end if

  // set texunit # & scale
  mTexture2DRectTonemapProgram.bind();
  mTexture2DRectTonemapProgram.setUniform1f("a", 0.18f);
  mTexture2DRectTonemapProgram.setUniform1i("texture", 0);
  mTexture2DRectTonemapProgram.unbind();

  source = "\n\
    uniform float scale;\n\
    uniform float gamma;\n\
    void main(void)\n\
    {\n\
      gl_FragColor.rgb = pow(scale * gl_Color.rgb, 1.0f / gamma);\n\
      if(gl_FragColor.r != gl_FragColor.r ||\n\
         gl_FragColor.g != gl_FragColor.g ||\n\
         gl_FragColor.b != gl_FragColor.b)\n\
      {\n\
        gl_FragColor.rgb = vec3(1,0,0);\n\
      }\n\
    }";
  mPassthroughGammaShader.create(GL_FRAGMENT_SHADER, source.c_str());
  if(!mPassthroughGammaProgram.create(0,0,mPassthroughGammaShader))
  {
    std::cerr << "CommonViewer::reloadShaders(): Problem creating mPassthroughGammaProgram." << std::endl;
    std::cerr << mPassthroughGammaProgram << std::endl;
  } // end if

  // set texunit # & default state
  mPassthroughGammaProgram.bind();
  mPassthroughGammaProgram.setUniform1f("scale", 1.0);
  mPassthroughGammaProgram.setUniform1f("gamma", 1.0);
  mPassthroughGammaProgram.unbind();
} // end CommonViewer::reloadShaders()

template<typename Parent, typename KeyEvent, typename StringType, typename VectorType>
  std::string CommonViewer<Parent,KeyEvent,StringType,VectorType>
    ::getOpenFileName(const char *prompt,
                      const char *path,
                      const char *desc)
{
  std::string result;
#ifdef QT_VERSION
  // get the filename
  StringType s = QFileDialog::getOpenFileName(this,
                                              prompt,
                                              path,
                                              desc);
  result = std::string(s.toAscii());
#else
  std::cout << prompt << std::endl;
  std::cin >> result;
#endif // QT_VERSION

  return result;
} // end CommonViewer::getOpenFileName()

template<typename Parent, typename KeyEvent, typename StringType, typename VectorType>
  void CommonViewer<Parent,KeyEvent,StringType,VectorType>
    ::displayMessage(const std::string &message,
                     int delay)
{
  Parent::displayMessage(StringType(message.c_str()),delay);
} // end CommonViewer::displayMessage()

