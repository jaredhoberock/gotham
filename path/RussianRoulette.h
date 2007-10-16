/*! \file RussianRoulette.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         for computing Russian roulette probabilities.
 */

#ifndef RUSSIAN_ROULETTE_H
#define RUSSIAN_ROULETTE_H

class Spectrum;
class DifferentialGeometry;
class Vector;

class RussianRoulette
{
  public:
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const = 0;
}; // end RussianRoulette

// always ballin'
class AlwaysRoulette
  : public RussianRoulette
{
  public:
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end AlwaysRoulette

class ConstantRoulette
  : public RussianRoulette
{
  public:
    ConstantRoulette(const float continueProbability = 0.3f);

    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;

    float getContinueProbability(void) const;

  protected:
    float mContinueProbability;
}; // end ConstantRoulette

class LuminanceRoulette
  : public RussianRoulette
{
  public:
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end LuminanceRoulette

class MaxOverSpectrumRoulette
  : public RussianRoulette
{
  public:
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end MaxOverSpectrumRoulette

class OnlyAfterDeltaRoulette
  : public RussianRoulette
{
  public:
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end OnlyAfterDeltaRoulette

class ConstantAndAlwaysAfterDeltaRoulette
  : public ConstantRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ConstantRoulette Parent;

    ConstantAndAlwaysAfterDeltaRoulette(const float continueProbability = 0.3f);

    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end ConstantAndAlwaysAfterDeltaRoulette

class KelemenRoulette
  : public ConstantRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ConstantRoulette Parent;

    KelemenRoulette(const float continueProbability = 0.3f);

    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end KelemenRoulette

#endif // RUSSIAN_ROULETTE_H

