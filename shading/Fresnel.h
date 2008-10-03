/*! \file Fresnel.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         abstracting the Fresnel functions.
 */

#pragma once
#include "../include/detail/Spectrum.h"

class FunctionAllocator;

class Fresnel
{
  public:
    /*! Null destructor does nothing.
     */
    inline virtual ~Fresnel(void);

    virtual Spectrum evaluate(const float cosi) const = 0;

    /*! This function computes the fresnel function for dielectrics.
     *  \param etai The index of refraction on the incident side of the
     *              interface.
     *  \param etat The index of refraction on the transmitted side of the
     *              interface.
     *  \param cosi The cosine of the angle between the incident direction
     *              and the normal at the interface.
     *  \param cost The cosine of the angle bewteen the transmitted direction
     *              and the normal at the interface.
     *  \return The fresnel transmittance.
     */
    inline static float dielectric(const float etai, const float etat,
                                   const float cosi, const float cost);

    inline static float dielectric(const float etai, const float etat,
                                   float cosi);

    inline static Spectrum conductor(const float cosi, const Spectrum &eta,
                                     const Spectrum &k);

    inline static Spectrum approximateEta(const Spectrum &r);

    inline static Spectrum approximateAbsorbance(const Spectrum &r);

    /*! Overload the new operator.
     */
    void *operator new(size_t size, FunctionAllocator &alloc);
}; // end Fresnel

class FresnelDielectric
  : public Fresnel
{
  public:
    FresnelDielectric(const float ei, const float et);

    Spectrum evaluate(const float cosi) const;

    Spectrum evaluate(const float cosi, const float cost) const;

    float mEtai, mEtat;
}; // end FresnelDielectric

class FresnelConductor
  : public Fresnel
{
  public:
    FresnelConductor(const Spectrum &e, const Spectrum &a);

    Spectrum evaluate(const float cosi) const;

  protected:
    Spectrum mEta, mAbsorbance;
}; // end FresnelConductor

#include "Fresnel.inl"

