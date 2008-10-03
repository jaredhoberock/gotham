/*! \file Texture.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Texture class.
 */

#ifdef WIN32
#define OPENEXR_DLL
#define NOMINMAX
#endif // WIN32

#include "Texture.h" 
#include <numeric>

#include <halfLimits.h>
#include <ImfRgbaFile.h>
#include <ImfRgba.h>
#include <ImfStringAttribute.h>
#include <ImfHeader.h>
#include <ImfArray.h>

#undef BOOST_NO_EXCEPTIONS

#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/gil/extension/io/jpeg_dynamic_io.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <boost/mpl/vector.hpp>

Texture
  ::Texture(void)
    :Parent(1,1)
{
  Parent::raster(0,0) = Spectrum::white();
} // end Texture::Texture()

Texture
  ::Texture(const size_t w, const size_t h, const Spectrum *pixels)
    :Parent(w,h)
{
  const Spectrum *src = pixels;
  for(iterator p = begin(); p != end(); ++p, ++src)
  {
    *p = *src;
  } // end for i
} // end Texture::Texture()

// create a gamma-inversion lookup table for
// gamma = 2.2 and unsigned chars
template<typename T>
  inline static float gammaInversion(const T x)
{
  static std::vector<float> lut(std::numeric_limits<T>::max() + 1);

  static bool firstTime = true;
  if(firstTime)
  {
    for(size_t i = 0; i != std::numeric_limits<T>::max() + 1; ++i)
    {
      // convert uchar to [0,1]
      float f = static_cast<float>(i) / (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
    
      // gamma correction:
      // gammaCorrect = rgb^(1.0/gamma)
      
      // inversion:
      // rgb = gammaCorrect^(gamma)
    
      // assume gamma = 2.2
    
      // invert x
      lut[i] = powf(f, 2.2f);
    } // end for i
    
    firstTime = false;
  } // end if

  return lut[x];
} // end gammaInversion()

template<typename View>
  static void readGrayImg(const View &v,
                          Array2<Spectrum> &tex)
{
  using namespace boost::gil;
  
  // now resize ourself
  tex.resize(v.width(),v.height());

  float factor = 1.0f / channel_traits<typename channel_type<View>::type>::max_value();

  // convert unsigned characters to gamma-corrected float in [0,1]
  Spectrum *dst = &tex.raster(0,0);
  for(size_t y = 0; y < v.height(); ++y)
  {
    // flip the image because gil images are stored ass-backwards
    size_t x = 0;
    for(typename View::x_iterator src = v.row_begin(v.height()-1-y);
        x < v.width();
        ++src, ++x, ++dst)
    {
      //(*dst).x = factor * static_cast<float>(*src);
      //(*dst).y = factor * static_cast<float>(*src);
      //(*dst).z = factor * static_cast<float>(*src);
      (*dst).x = gammaInversion(*src);
      (*dst).y = gammaInversion(*src);
      (*dst).z = gammaInversion(*src);
    } // end for src, x
  } // end for y
} // end readImg()

template<typename View>
  static void readRgbImg(const View &v,
                         Array2<Spectrum> &tex)
{
  using namespace boost::gil;

  // now resize ourself
  tex.resize(v.width(),v.height());

  Spectrum *dst = &tex.raster(0,0);
  for(size_t y = 0; y < v.height(); ++y)
  {
    // flip the image because gil images are stored ass-backwards
    typename View::x_iterator src = v.row_begin(v.height()-1-y);

    size_t x = 0;
    for(typename View::x_iterator src = v.row_begin(v.height()-1-y);
        x < v.width();
        ++src, ++x, ++dst)
    {
      (*dst).x = gammaInversion((*src)[0]);
      (*dst).y = gammaInversion((*src)[1]);
      (*dst).z = gammaInversion((*src)[2]);
    } // end for src, x
  } // end for y
} // end readImg()

static void readJPG(const char *filename,
                    Array2<Spectrum> &tex)
{
  // try different pixel formats
  //std::cerr << "readJPG(): reading file." << std::endl;
  try
  {
    boost::gil::rgb8_image_t img;
    boost::gil::jpeg_read_image(filename, img);
    //std::cerr << "readJPG(): done." << std::endl;
    readRgbImg(const_view(img), tex);
  } // end try
  catch(...)
  {
    boost::gil::gray8_image_t img;
    boost::gil::jpeg_read_image(filename, img);
    //std::cerr << "readJPG(): done." << std::endl;
    readGrayImg(const_view(img), tex);
  } // end catch
} // end readJPG()

static void readPNG(const char *filename,
                    Array2<Spectrum> &tex)
{
  boost::gil::rgba8_image_t img;
  boost::gil::png_read_image(filename, img);

  readRgbImg(const_view(img), tex);
} // end readJPG()

static void readEXR(const char *filename,
                    Array2<Spectrum> &image)
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

void Texture
  ::load(const char *filename)
{
  try
  {
    readEXR(filename, *this);
  } // end try
  catch(...)
  {
    try
    {
      readPNG(filename, *this);
    } // end try
    catch(std::ios_base::failure e)
    {
      // XXX for some reason, this isn't throwing an exception
      //     it just exit()s
      readJPG(filename, *this);
    } // end catch
  } // end catch
} // end Texture::load()

Texture
  ::Texture(const char *filename)
{
  load(filename);
} // end Texture::Texture()

const Spectrum &Texture
  ::texRect(const size_t x,
            const size_t y) const
{
  // clamp to dim - 1
  return Parent::raster(std::min<size_t>(x, mDimensions[0]-1),
                        std::min<size_t>(y, mDimensions[1]-1));
} // end Texture::texRect()

const Spectrum &Texture
  ::tex2D(const float u,
          const float v) const
{
  int x = static_cast<int>(u * getDimensions()[0]);
  int y = static_cast<int>(v * getDimensions()[1]);

  // clamp to 0
  return texRect(std::max<int>(0,x),
                 std::max<int>(0,y));
} // end Texture::tex2D()

