/*! \file Mappings.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         which provides functions for mapping
 *         the unit square to various familiar
 *         manifolds.
 */

#ifndef MAPPINGS_H
#define MAPPINGS_H

#include "Vector.h"

class Mappings
{
  public:
    /*! This method maps a point in the unit square to a point on the unit
     *  sphere situated around the z-axis.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param xAxis A unit vector parallel to the x-axis of
     *               defining the frame of the sphere of interest.
     *  \param yAxis A unit vector parallel to the y-axis of
     *               defining the frame of the sphere of interest.
     *  \param zAxis A unit vector parallel to the z-axis of
     *               defining the frame of the sphere of interest.
     *  \param w The point on the unit sphere is returned here.
     *  \param pdf The value of the pdf at p is returned here.
     *  \note xAxis, yAxis, and zAxis are assumed to define an
     *        orthonormal basis.
     */
    inline static void unitSquareToSphere(const float u0,
                                          const float u1,
                                          const Vector3 &xAxis,
                                          const Vector3 &yAxis,
                                          const Vector3 &zAxis,
                                          Vector3 &w,
                                          float &pdf);

    /*! This method maps a point in the unit square
     *  to a point on the unit hemisphere situated around
     *  the z-axis.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param xAxis A unit vector parallel to the x-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param yAxis A unit vector parallel to the y-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param zAxis A unit vector parallel to the z-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param w The point on the unit hemisphere is returned here.
     *  \param pdf The value of the pdf at p is returned here.
     *  \note xAxis, yAxis, and zAxis are assumed to define an
     *        orthonormal basis.
     */
    inline static void unitSquareToHemisphere(const float u0,
                                              const float u1,
                                              const Vector3 &xAxis,
                                              const Vector3 &yAxis,
                                              const Vector3 &zAxis,
                                              Vector3 &w,
                                              float &pdf);

    /*! This method maps a point in the unit square to a point
     *  on the cosine-weighted unit hemisphere situated around
     *  a vector identified with the z-axis.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param xAxis A unit vector parallel to the x-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param yAxis A unit vector parallel to the y-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param zAxis A unit vector parallel to the z-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param w The point on the unit hemisphere is returned here.
     *  \param pdf The value of the pdf at p is returned here.
     *  \note xAxis, yAxis, and zAxis are assumed to define an
     *        orthonormal basis.
     */
    inline static void unitSquareToCosineHemisphere(const float u0,
                                                    const float u1,
                                                    const Vector &xAxis,
                                                    const Vector &yAxis,
                                                    const Vector &zAxis,
                                                    Vector &w,
                                                    float &pdf);

    /*! This method maps a point in the unit square to a point
     *  on a Phong lobe.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param xAxis A unit vector parallel to the x-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param yAxis A unit vector parallel to the y-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param zAxis A unit vector parallel to the z-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param w The point on the unit hemisphere is returned here.
     *  \param pdf The value of the pdf at p is returned here.
     *  \note xAxis, yAxis, and zAxis are assumed to define an
     *        orthonormal basis.
     */
    inline static void unitSquareToPhongLobe(const float u0,
                                             const float u1,
                                             const Vector &r,
                                             const float exponent,
                                             const Vector &xAxis,
                                             const Vector &yAxis,
                                             const Vector &zAxis,
                                             Vector &w,
                                             float &pdf);

    /*! This method maps a point in the unit square to a point on an
     *  anisotropic Phong lobe.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param nu The Phong exponent in the u direction.
     *  \param nv The Phong exponent in the v direction.
     *  \param xAxis A unit vector parallel to the x-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param yAxis A unit vector parallel to the y-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param zAxis A unit vector parallel to the z-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param w The point on the unit hemisphere is returned here.
     *  \param pdf The value of the pdf at p is returned here.
     *  \note xAxis, yAxis, and zAxis are assumed to define an
     *        orthonormal basis.
     */
    inline static void unitSquareToAnisotropicPhongLobe(const float u0,
                                                        const float u1,
                                                        const float nu,
                                                        const float nv,
                                                        const Vector &xAxis,
                                                        const Vector &yAxis,
                                                        const Vector &zAxis,
                                                        Vector &w,
                                                        float &pdf);

    /*! This method evaluates the pdf of the unitSquareToAnisotropicPhongLobe
     *  function.
     *  \param w A vector on the anisotropic lobe.
     *  \param nu The first exponent of the anisotropic lobe.
     *  \param nv The second exponent of the anisotropic lobe.
     *  \param xAxis A vector defining the tangent direction orthogonal to zAxis.
     *  \param yAxis A vector defining the binormal direction orthogonal to zAxis.
     *  \param zAxis A vector defining the radial axis of the lobe.
     *  \return The pdf of choosine w from the anisotropic lobe
     *          centered about r.
     */
    inline static float evaluateAnisotropicPhongLobePdf(const Vector &w,
                                                        const float nu,
                                                        const float nv,
                                                        const Vector &xAxis,
                                                        const Vector &yAxis,
                                                        const Vector &zAxis);

    /*! This method evaluates the pdf of the unitSquareToPhongLobe
     *  function.
     *  \param w A vector on the Phong lobe.
     *  \param r A vector defining the radial axis of the Phong lobe.
     *  \param exponent The exponent of the Phong lobe.
     *  \return The pdf of choosine w from the Phong lobe
     *          centered about r.
     */
    inline static float evaluatePhongLobePdf(const Vector &w,
                                             const Vector &r,
                                             const float exponent);

    /*! This method evaluates the pdf of the unitSquareToCosineHemisphere
     *  function.
     *  \param w A vector on the cosine-weighted hemisphere.
     *  \param zAxis A unit vector parallel to the z-axis of the
     *               frame defining the hemisphere of interest.
     *  \return The pdf of choosine w from the cosine-weighted hemisphere
     *          centered about zAxis.
     */
    inline static float evaluateCosineHemispherePdf(const Vector &w,
                                                    const Vector &zAxis);

    /*! This method maps a direction on the cosine hemisphere to a point
     *  in the unit square to a point.
     *  \param w A direction on the cosine-weighted hemisphere.
     *  \param xAxis A unit vector parallel to the x-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param yAxis A unit vector parallel to the y-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param zAxis A unit vector parallel to the z-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param u0 A real number in [0,1) is returned here if w lies on the
     *            cosine-weighted hemisphere about zAxis.
     *  \param u1 A second real number in [0,1) is returned here if w lies on
     *            the cosine-weighted hemisphere about zAxis.
     *  \note xAxis, yAxis, and zAxis are assumed to define an
     *        orthonormal basis.
     */
    inline static void cosineHemisphereToUnitSquare(const Vector &w,
                                                    const Vector &xAxis,
                                                    const Vector &yAxis,
                                                    const Vector &zAxis,
                                                    float &u0,
                                                    float &u1);
}; // end Mappings

#include "Mappings.inl"

#endif // MAPPINGS_H

