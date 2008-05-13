/*! \file PerspectiveSensor.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a SensorFunction
 *         implementing a perspective camera.
 */

#ifndef PERSPECTIVE_SENSOR_H
#define PERSPECTIVE_SENSOR_H

#include "ScatteringDistributionFunction.h"
#include "functions/PerspectiveSensorBase.h"

class PerspectiveSensor
  : public ScatteringDistributionFunction,
    public PerspectiveSensorBase<Vector,Spectrum,DifferentialGeometry>
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef ScatteringDistributionFunction Parent0;
    
    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef PerspectiveSensorBase<Vector,Spectrum,DifferentialGeometry> Parent1;

    /*! Null constructor does nothing.
     */
    PerspectiveSensor(void);

    /*! Constructor accepts an aspect ratio, field of view,
     *  and window origin.
     *  \param response A constant response to radiance.
     *  \param aspect Sets mAspectRatio.
     *  \param origin Sets mWindowOrigin.
     */
    PerspectiveSensor(const Spectrum &response,
                      const float aspect,
                      const Point &origin);

    /*! This method evaluates this PerspectiveSensor's sensor response
     *  to radiance in a given sensing direction.
     *  \param ws A vector pointing towards the incoming direction.
     *  \param dg The DifferentialGeometry at the Point on the sensor.
     *  \return The sensor response to incoming radiance from ws.
     */
    using Parent0::evaluate;
    virtual Spectrum evaluate(const Vector &ws,
                              const DifferentialGeometry &dg) const;
    
    /*! This method which samples this PerspectiveSensor's sensor window
     *  given a DifferentialGeometry, and two numbers.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param ws The direction of sensing is returned here.
     *  \param pdf The value of the solid angle pdf at (u0,u1) is
     *             returned here.
     *  \param delta This is set to false.
     *  \return The sensor response to incoming radiance from ws
     *          is returned here.
     */
    using Parent0::sample;
    virtual Spectrum sample(const DifferentialGeometry &dg,
                            const float u0,
                            const float u1,
                            const float u2,
                            Vector &ws,
                            float &pdf,
                            bool &delta) const;

    /*! This method inverts this PerspectiveSensor's mapping
     *  from a direction to the unit square.
     *  \param w The direction of interest.
     *  \param dg the DifferentialGeometry at the Point of interest.
     *  \param u0 The first coordinate of the corresponding point in the unit
     *            square is returned here.
     *  \param u1 The second coordinate of the corresponding point in the unit
     *            square is returned here.
     */
    virtual void invert(const Vector &w,
                        const DifferentialGeometry &dg,
                        float &u0,
                        float &u1) const;

    /*! This method evaluates the solid angle pdf of the
     *  sensing direction of interest.
     *  \param ws The sensing direction of interest.
     *  \param dg The DifferentialGeometry at the Point of interest.
     *  \return The solid angle pdf of ws.
     */
    using Parent0::evaluatePdf;
    virtual float evaluatePdf(const Vector &ws,
                              const DifferentialGeometry &dg) const;
}; // end PerspectiveSensor

#endif // PERSPECTIVE_SENSOR_H

