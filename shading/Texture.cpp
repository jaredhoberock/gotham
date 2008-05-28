/*! \file Texture.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Texture class.
 */

#include "Texture.h" 

#ifdef WIN32
#define OPENEXR_DLL
#endif // WIN32

#include <halfLimits.h>
#include <ImfRgbaFile.h>
#include <ImfRgba.h>
#include <ImfStringAttribute.h>
#include <ImfHeader.h>
#include <ImfArray.h>

Texture
  ::Texture(void)
    :Parent(1,1)
{
  raster(0,0) = Spectrum::white();
} // end Texture::Texture()

Texture
  ::Texture(const size_t w, const size_t h)
    :Parent(w,h)
{
  ;
} // end Texture::Texture()

static void readEXR(const char *filename,
                    Texture &image)
{
  Imf::RgbaInputFile file(filename);

  Imath::Box2i dw = file.dataWindow();

  unsigned int w = dw.max.x - dw.min.x + 1;
  unsigned int h = dw.max.y - dw.min.y + 1;

  Imf::Array2D<Imf::Rgba> pixels;
  pixels.resizeErase(h, w);

  file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * w, 1, w);
  file.readPixels(dw.min.y, dw.max.y);

  // now resize ourself
  image.resize(w,h);

  for(size_t y = 0; y < w; ++y)
  {
    for(size_t x = 0; x < h; ++x)
    {
      // flip the image because EXR is stored ass-backwards
      image.raster(x,y).x = pixels[h-1-y][x].r;
      image.raster(x,y).y = pixels[h-1-y][x].g;
      image.raster(x,y).z = pixels[h-1-y][x].b;
    } // end for
  } // end for
} // end readEXR()

Texture
  ::Texture(const char *filename)
{
  readEXR(filename, *this);
} // end Texture::Texture()

