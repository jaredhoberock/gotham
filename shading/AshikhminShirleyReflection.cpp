/*! \file AshikhminShirleyReflection.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of AshikhminShirleyReflection class.
 */

#include "AshikhminShirleyReflection.h"
#include "../geometry/Mappings.h"

AshikhminShirleyReflection
  ::AshikhminShirleyReflection(const Spectrum &r,
                               const float eta,
                               const float uExponent,
                               const float vExponent)
    :Parent0(),
     Parent1(r,eta,uExponent,vExponent)
{
  ;
} // end AshikhminShirleyReflection::AshikhminShirleyReflection()

AshikhminShirleyReflection
  ::AshikhminShirleyReflection(const Spectrum &r,
                               const float etai,
                               const float etat,
                               const float uExponent,
                               const float vExponent)
    :Parent0(),
     Parent1(r,etai,etat,uExponent,vExponent)
{
  ;
} // end AshikhminShirleyReflection::AshikhminShirleyReflection()

Spectrum AshikhminShirleyReflection
  ::sample(const Vector &wo,
           const DifferentialGeometry &dg,
           const float u0,
           const float u1,
           const float u2,
           Vector &wi,
           float &pdf,
           bool &delta,
           ComponentIndex &index) const
{
  return Parent1::sample(wo,
                         dg.getTangent(),
                         dg.getBinormal(),
                         dg.getNormal(),
                         u0,u1,u2,
                         wi,pdf,delta,index);
} // end AshikhminShirleyReflection::sample()

Spectrum AshikhminShirleyReflection
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi,
             const bool delta,
             const ComponentIndex component,
             float &pdf) const
{
  pdf = evaluatePdf(wo,dg,wi);
  return Parent1::evaluate(wo,dg.getTangent(),dg.getBinormal(),dg.getNormal(),wi);
} // end AshikhminShirleyReflection::evaluate()

Spectrum AshikhminShirleyReflection
  ::evaluate(const Vector &wo,
             const DifferentialGeometry &dg,
             const Vector &wi) const
{
  return Parent1::evaluate(wo,dg.getTangent(),dg.getBinormal(),dg.getNormal(),wi);
} // end AshikhminShirleyReflection::evaluate()

float AshikhminShirleyReflection
  ::evaluatePdf(const Vector &wo,
                const DifferentialGeometry &dg,
                const Vector &wi) const
{
  return Parent1::evaluatePdf(wo,dg.getTangent(),dg.getBinormal(),dg.getNormal(),wi);
} // end AshikhminShirleyReflection::evaluatePdf()

