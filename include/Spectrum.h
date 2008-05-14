/*! \file Spectrum.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class representing a spectral distribution function.
 */

#ifndef SPECTRUM_H
#define SPECTRUM_H

#include "gpcpu/Vector.h"
//#include <gpcpu/floatmxn.h>

class Spectrum
  : public gpcpu::float3
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef gpcpu::float3 Parent;

    /*! Null constructor calls the Parent.
     */
    inline Spectrum(void);

    /*! Copy constructor accepts a Parent.
     *  \param p A Parent to copy from.
     */
    inline Spectrum(const Parent &p);

    /*! Constructor takes three scalars.
     *  \param s0 The first scalar.
     *  \param s1 The second scalar.
     *  \param s2 The third scalar.
     */
    inline Spectrum(const float s0, const float s1, const float s2);

    inline float luminance(void) const;

    /*! This method forces this Spectrum's luminance() method to
     *  return the given value.
     *  \param l The value of interest.
     */
    inline void setLuminance(const float l);

    ///*! This method interprets this Spectrum as an XYZ coordinate
    // *  and returns its representation in RGB.
    // *  \return The RGB representation of this XYZ coordinate.
    // */
    //inline Spectrum toRGB(void) const;

    ///*! This method interprets this Spectrum as an RGB coordinate
    // *  and returns its representation in XYZ.
    // *  \return The XYZ representation of this RGB coordinate.
    // */
    //inline Spectrum toXYZ(void) const;

    ///*! This method interprets this Spectrum as an XYZ coordinate
    // *  and returns its representation in xy luminance.
    // *  \return the xy luminance representation of this XYZ coordinate.
    // */
    //inline Spectrum toXYLuminance(void) const;

    ///*! This method interprets this Spectrum as an Yxy coordinate
    // *  and returns its representation in in XYZ.
    // *  \return the XYZ representation of this Yxy coordinate.
    // */
    //inline Spectrum fromYxyToXYZ(void) const;

    /*! This static method returns a zero Spectrum.
     *  \return Spectrum(0,0,0)
     */
    inline static Spectrum black(void);

    /*! This static method returns a full Spectrum.
     *  \return Spectrum(1,1,1)
     */
    inline static Spectrum white(void);

    /*! This method returns whether or not this Spectrum
     *  is completely zero.
     *  \return true if this Spectrum is close enough to zero
     *               to be considered black; false, otherwise.
     */
    inline bool isBlack(void) const;

  protected:
    /*! Color space conversion matrices */
    //static const gpcpu::float3x3 XYZ_TO_RGB;
    //static const gpcpu::float3x3 RGB_TO_XYZ;
}; // end Spectrum

#include "Spectrum.inl"

#endif // SPECTRUM_H

