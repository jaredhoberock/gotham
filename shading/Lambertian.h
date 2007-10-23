/*! \file Lambertian.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Lambertian
 *         scattering function.
 */

#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "ScatteringDistributionFunction.h"

class Lambertian
  : public ScatteringDistributionFunction
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent;

    /*! Constructor accepts an albedo.
     *  \param albedo Sets mAlbedo.
     */
    Lambertian(const Spectrum &albedo);

    /*! This method computes Lambertian scattering.
     *  \param wo The scattering direction.
     *  \param dg The DifferentialGeometry at the Point to be shaded.
     *  \param wi The incoming direction.
     *  \return mAlbedo / PI if wi & wo are in the same hemisphere;
     *          0, otherwise.
     */
    using Parent::evaluate;
    Spectrum evaluate(const Vector3 &wo,
                      const DifferentialGeometry &dg,
                      const Vector3 &wi) const;

    /*! This method computes the value of this Lambertian and its pdf.
     *  \param wo A Vector pointing towards the direction of scattering.
     *  \param dg The DifferentialGeometry at the scattering Point of interest.
     *  \param wi A Vector pointing towards the direction of incidence.
     *  \param delta Ignored.  Lambertian is not a delta function.
     *  \param component Ignored.  Lambertian has only one component.
     *  \param pdf The value of this ScatteringDistributionFunction's pdf is returned here.
     *  \return mAlbedo / PI if wi & wo are in the same hemisphere; 0, otherwise.
     */
    Spectrum evaluate(const Vector &wo,
                      const DifferentialGeometry &dg,
                      const Vector &wi,
                      const bool delta,
                      const ComponentIndex component,
                      float &pdf) const;

  protected:
    Spectrum mAlbedo;

    /*! mAlbedo / PI
     */
    Spectrum mAlbedoOverPi;
}; // end Lambertian

#endif // LAMBERTIAN_H

