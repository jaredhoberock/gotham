/*! \file Normal.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a vector class abstracting a normal vector.
 */

#ifndef NORMAL_H
#define NORMAL_H

#include <gpcpu/Vector.h>

/*! \class Normal
 *  \brief A Normal is a float3.
 */
class Normal
  : public float3
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef float3 Parent;

    /*! \fn Normal
     *  \brief Null constructor calls the Parent.
     */
    inline Normal(void):Parent(){;}

    /*! \fn Normal
     *  \brief Constructor accepts a float3.
     */
    inline Normal(const float3 &n):Parent(n){;}

    /*! \fn Normal
     *  \brief Constructor accepts three elements of the Normal.
     */
    inline Normal(const float &nx, const float &ny, const float &nz)
      :Parent(nx,ny,nz){;}

    /*! \fn sameHemisphere
     *  \brief This method returns a Normal parallel to this Normal
     *         in the same hemisphere as the given vector.
     *  \param v The vector of interest.
     *  \return A Normal parallel to this Normal in the same hemisphere
     *          as v.
     */
    inline Normal sameHemisphere(const float3 &v) const
    {
      if(dot(v) < 0) return -*this;
      return *this;
    } // end sameHemisphere()
}; // end class Normal

#endif // NORMAL_H

