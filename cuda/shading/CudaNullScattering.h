/*! \file CudaNullScattering.h
 *  \author Jared Hoberock
 *  \brief Interface to a CudaScatteringDistributionFunction that
 *         returns null scattering events.
 */

#pragma once

// XXX hack
#define inline inline __host__ __device__
#include "../geometry/CudaDifferentialGeometry.h"
#undef inline

class CudaNullScattering
{
  public:
    /*! This method evaluates a bidirectional scattering event in an outgoing
     *  direction of interest given a direction of incidence.
     *  \param wo A vector pointing towards the outgoing scattered direction
     *  \param dg The DifferentialGeometry at the Point to be shaded.
     *  \param wi A vector pointing towards the incoming direction.
     *  \return The bidirectional scattering toward direction wo from wi.
     *  \note This method returns (0,0,0).
     */
    inline __host__ __device__ float3 evaluate(const float3 &wo,
                                               const CudaDifferentialGeometry &dg,
                                               const float3 &wi) const;

    /*! This method evaluates a unidirectional scattering event in an outgoing
     *  direction of interest.
     *  \param wo A vector pointing towards the outgoing scattered direction
     *  \param dg The DifferentialGeometry at the Point to be shaded.
     *  \return The bidirectional scattering toward direction wo from wi.
     *  \note This method returns (0,0,0).
     */
    inline __host__ __device__ float3 evaluate(const float3 &wo,
                                               const CudaDifferentialGeometry &dg) const;
}; // end CudaNullScattering

#include "CudaNullScattering.inl"

