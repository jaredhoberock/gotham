/*! \file Ray.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Ray class.
 */

#include "Ray.h"
#include <limits>

/*! Initialize these here, because C++ is jerks. */
const float Ray::RAY_EPSILON = 1e-3f;
const float Ray::RAY_INFINITY = std::numeric_limits<float>::infinity();

