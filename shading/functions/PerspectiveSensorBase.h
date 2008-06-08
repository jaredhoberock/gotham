/*! \file PerspectiveSensorBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to
 *         a simple class encapsulating
 *         a PerspectiveSensor function.
 */

#pragma once

template<typename V3, typename S3>
  class PerspectiveSensorBase
{
  public:
    typedef V3 Point;
    typedef V3 Vector;
    typedef S3 Spectrum;

    /*! Null constructor does nothing.
     */
    inline PerspectiveSensorBase(void);

    /*! Constructor accepts an aspect ratio, field of view,
     *  and window origin.
     *  \param response A constant response to radiance.
     *  \param aspect Sets mAspectRatio.
     *  \param origin Sets mWindowOrigin.
     */
    inline PerspectiveSensorBase(const Spectrum &response,
                                 const float aspect,
                                 const Point &origin);

    /*! This method sets the members of this PerspectiveSensor.
     *  \param aspect Sets mAspectRatio.
     *  \param origin Sets mWindowOrigin.
     */
    inline void set(const float aspect, const Point &origin);

    /*! This method evaluates this PerspectiveSensor's sensor response
     *  to radiance in a given sensing direction.
     *  \param ws A vector pointing towards the incoming direction.
     *  \param point The position of the point of interest.
     *  \param tangent The tangent vector at the point of interest.
     *  \param binormal The binormal vector at the point of interest.
     *  \param normal The normal vector at the point of interest.
     *  \return The sensor response to incoming radiance from ws.
     */
    inline Spectrum evaluate(const Vector &ws,
                             const Point &point,
                             const Vector &tangent,
                             const Vector &binormal,
                             const Vector &normal) const;

    /*! This method which samples this PerspectiveSensor's sensor window
     *  given a DifferentialGeometry, and two numbers.
     *  \param point The position on the sensor.
     *  \param tangent The tangent direction at point.
     *  \param binormal The binormal direction at point.
     *  \param normal The normal direction at point.
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
    inline Spectrum sample(const Point &point,
                           const Vector &tangent,
                           const Vector &binormal,
                           const Vector &normal,
                           const float u0,
                           const float u1,
                           const float u2,
                           Vector &ws,
                           float &pdf,
                           bool &delta) const;

    /*! This method which samples this PerspectiveSensor's sensor window
     *  given a DifferentialGeometry, and two numbers.
     *  \param point The position on the sensor.
     *  \param tangent The tangent direction at point.
     *  \param binormal The binormal direction at point.
     *  \param normal The normal direction at point.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param ws The direction of sensing is returned here.
     *  \param pdf The value of the solid angle pdf at (u0,u1) is
     *             returned here.
     *  \param delta This is set to false.
     *  \param component This is set to 0.
     *  \return The sensor response to incoming radiance from ws
     *          is returned here.
     */
    inline Spectrum sample(const Point &point,
                           const Vector &tangent,
                           const Vector &binormal,
                           const Vector &normal,
                           const float u0,
                           const float u1,
                           const float u2,
                           Vector &ws,
                           float &pdf,
                           bool &delta,
                           unsigned int &component) const;

    /*! This method inverts this PerspectiveSensor's mapping
     *  from a direction to the unit square.
     *  \param w The direction of interest.
     *  \param point The position of the point of interest.
     *  \param tangent The tangent at the Point of interest.
     *  \param binormal The binormal at the Point of interest.
     *  \param normal The normal at the Point of interest.
     *  \param u0 The first coordinate of the corresponding point in the unit
     *            square is returned here.
     *  \param u1 The second coordinate of the corresponding point in the unit
     *            square is returned here.
     */
    inline void invert(const Vector &w,
                       const Point &point,
                       const Vector &tangent,
                       const Vector &binormal,
                       const Vector &normal,
                       float &u0,
                       float &u1) const;

    /*! This method evaluates the solid angle pdf of the
     *  sensing direction of interest.
     *  \param ws The sensing direction of interest.
     *  \param point The position of the point of interest.
     *  \param tangent The tangent at the Point of interest.
     *  \param binormal The binormal at the Point of interest.
     *  \param normal The normal at the Point of interest.
     *  \return The solid angle pdf of ws.
     */
    inline float evaluatePdf(const Vector &ws,
                             const Point &point,
                             const Vector &tangent,
                             const Vector &binormal,
                             const Vector &normal) const;

  protected:
    /*! This method samples the surface area of the window.
     *  \param u A real number in [0,1).
     *  \param v A real number in [0,1).
     *  \param xAxis A unit vector parallel to the sensor's x-axis.
     *  \param yAxis A unit vector parallel to the sensor's y-axis.
     *  \param zAxis A unit vector parallel to the sensor's z-axis.
     *  \param p A Point uniformly sampled from the surface area of the
     *           sensor window is returned here.
     *  \param pdf The surface area pdf at p is returned here.
     */
    inline void sampleWindow(const float u,
                             const float v,
                             const Vector &xAxis,
                             const Vector &yAxis,
                             const Vector &zAxis,
                             Point &p,
                             float &pdf) const;

    /*! A PerspectiveSensor has a vertical field of view expressed
     *  in radians.
     */
    float mFovY;

    /*! A PerspectiveSensor has an aspect ratio: width / height of the
     *  sensor window.
     */
    float mAspectRatio;

    /*! One over the surface area of the window.
     */
    float mInverseWindowSurfaceArea;

    /*! The origin (lower left hand corner) of the sensor window.
     */
    Point mWindowOrigin;

    /*! The response to radiance.
     */
    Spectrum mResponse;
}; // end PerspectiveSensorBase

#include "PerspectiveSensorBase.inl"

