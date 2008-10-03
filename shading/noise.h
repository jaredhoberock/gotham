/*! \file noise.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a function
 *         for generating Perlin noise.
 */

#ifndef NOISE_H
#define NOISE_H

#include "../include/detail/exportShading.h"

/*! This function takes the three coordinates of a Point
 *  and returns the value of the signed Perlin noise function
 *  at that Point.
 *  \param x The x-coordinate of the Point of interest.
 *  \param y The y-coordinate of the Point of interest.
 *  \param z The z-coordinate of the Point of interest.
 *  \return The value of the Perlin noise function at (x,y,z).
 */
float noise(float x, float y, float z);

#endif // NOISE_H

