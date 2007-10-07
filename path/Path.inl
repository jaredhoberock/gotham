/*! \file Path.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Path.inl.
 */

#include "Path.h"
#include "../primitives/Scene.h"

PathVertex
  ::PathVertex(void)
{
  ;
} // end PathVertex::PathVertex()

template<typename RNG>
  bool Path
    ::construct(const Scene *scene,
                const unsigned int eyeSubpathLength,
                const unsigned int lightSubpathLength,
                const float p0,
                const float p1,
                const float p2,
                RNG &rng)
{
  // clear
  mSubpathLengths = gpcpu::uint2(0,0);

  if(eyeSubpathLength > 0)
  {
    // insert a lens vertex
    insert(0, scene->getSensors(), false, rng(), rng(), rng(), rng());

    for(unsigned int i = 0; i < eyeSubpathLength -1; ++i)
    {
      bool scatter = true;
      float u0 = rng(), u1 = rng(), u2 = rng();
      if(i == 0)
      {
        scatter = false;
        u0 = p0;
        u1 = p1;
        u2 = p2;
      } // end if
     
      if(insert(i, scene, true, scatter, u0, u1, u2) == NULL_VERTEX) return false;
    } // end for i
  } // end if

  if(lightSubpathLength > 0)
  {
    // insert a light vertex
    insert(eyeSubpathLength + lightSubpathLength - 1, scene->getEmitters(), true, rng(), rng(), rng(), rng());

    for(unsigned int i = 0; i < lightSubpathLength - 1; ++i)
    {
      bool scatter = true;
      if(i == 0) scatter = false;

      if(insert(eyeSubpathLength + lightSubpathLength - i - 1,
                scene, false, scatter,
                rng(), rng(), rng()) == NULL_VERTEX) return false;
    } // end for i
  } // end if

  return true;;
} // end Path::construct()

