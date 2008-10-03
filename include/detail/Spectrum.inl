/*! \file Spectrum.inl
 *  \author Jared Hoberock *  \brief Inline file for Spectrum.h.
 */

#include "Spectrum.h"

Spectrum
  ::Spectrum(void)
    :Parent()
{
  ;
} // end Spectrum::Spectrum()

Spectrum
  ::Spectrum(const Parent &p)
    :Parent(p)
{
  ;
} // end Spectrum::Spectrum()

Spectrum
  ::Spectrum(const float s0, const float s1, const float s2)
    :Parent(s0,s1,s2)
{
  ;
} // end Spectrum::Spectrum()

float Spectrum
  ::luminance(void) const
{
  return 0.299f * (*this)[0] + 0.587f * (*this)[1] + 0.114f * (*this)[2];
} // end Spectrum::luminance()

void Spectrum
  ::setLuminance(const float l)
{
  float old = luminance();

  if(old > 0)
  {
    (*this)[0] /= (0.299f * old);
    (*this)[1] /= (0.587f * old);
    (*this)[2] /= (0.114f * old);

    (*this)[0] *= (0.299f * l);
    (*this)[1] *= (0.587f * l);
    (*this)[2] *= (0.114f * l);
  } // end if
  else
  {
    (*this)[0] = (0.299f * l);
    (*this)[1] = (0.587f * l);
    (*this)[2] = (0.114f * l);
  } // end else
} // end Spectrum::setLuminance()

//Spectrum Spectrum
//  ::toRGB(void) const
//{
//  //static const float3x3 XYZ_TO_RGB
//  //  = float3x3( 3.240479f, -1.537150f, -0.498535f,
//  //             -0.969256f,  1.875992f,  0.041556f,
//  //              0.055648f, -0.204043f,  1.057311f);
//
//  //return XYZ_TO_RGB * (*this);
//
//  Spectrum rgb;
//
//  rgb[0] = gpcpu::float3(3.240479f, -1.537150f, -0.498535f).dot(*this);
//  rgb[1] = gpcpu::float3(-0.969256f, 1.875992f,  0.041556f).dot(*this);
//  rgb[2] = gpcpu::float3(0.055648f, -0.204043f,  1.057311f).dot(*this);
//
//  return rgb;
//} // end Spectrum::toRGB()
//
//Spectrum Spectrum
//  ::toXYZ(void) const
//{
//  static const gpcpu::float3x3 XYZ_TO_RGB
//    = gpcpu::float3x3( 3.240479f, -1.537150f, -0.498535f,
//                      -0.969256f,  1.875992f,  0.041556f,
//                       0.055648f, -0.204043f,  1.057311f);
//
//  static const gpcpu::float3x3 RGB_TO_XYZ
//    = XYZ_TO_RGB.inverse();
//
//  return RGB_TO_XYZ * (*this);
//} // end Spectrum::toXYZ()
//
///*! from http://en.wikipedia.org/wiki/CIE_XYZ_color_space
// */
//Spectrum Spectrum
//  ::toXYLuminance(void) const
//{
//  float invDenom = 1.0f / sum();
//  float x = (*this)[0] * invDenom;
//  float y = (*this)[1] * invDenom;
//  float luminance = (*this)[1];
//
//  return Spectrum(x,y,luminance);
//} // end Spectrum::toXYLuminance()
//
//Spectrum Spectrum
//  ::fromYxyToXYZ(void) const
//{
//  Spectrum xyz;
//  const Spectrum &Yxy = *this;
//
//  float r = Yxy[0] / Yxy[2];
//
//  xyz[0] = Yxy[1] * r;
//  xyz[1] = Yxy[0];
//  xyz[2] = (1.0f - Yxy[1] - Yxy[2]) * r;
//
//  return xyz;
//} // end Spectrum::fromXYLuminanceToXYZ()

bool Spectrum
  ::isBlack(void) const
{
  //return norm2() < 0.00001f;
  return norm2() == 0;
} // end Spectrum::isBlack()

Spectrum Spectrum
  ::black(void)
{
  return Spectrum(0,0,0);
} // end Spectrum::black()

Spectrum Spectrum
  ::white(void)
{
  return Spectrum(1,1,1);
} // end Spectrum::white()

