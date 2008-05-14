/*! \file Quadratic.h
 *  \author Jared Hoberock
 *  \brief Defines the interface of a quadratic equation.
 */

#ifndef QUADRATIC_H
#define QUADRATIC_H

/*! \class Quadratic
 *  \brief Quadratic encapsulates the coefficients of a quadratic equation
 *         and provides a method for solving the roots.
 *         A quadratic equation, as abstracted by this class, takes the form: \n
 *         \f$q(x) = a x^2 + b x + c\f$
 *  \see http://mathworld.wolfram.com/QuadraticEquation.html
 */
class Quadratic
{
  public:
    /*! Constructor accepts three coefficients
     *  \param a The squared term coefficient.
     *  \param b The linear term coefficient.
     *  \param c The constant term coefficient.
     */
    inline Quadratic(const float a, const float b, const float c)
    {
      setCoefficients(a,b,c);
    } // end Quadratic::Quadratic()

    /*! This method sets the coefficients of the Quadratic.
     *  \param a The squared term coefficient.
     *  \param b The linear term coefficient.
     *  \param c The constant term coefficient.
     */
    inline void setCoefficients(const float a, const float b, const float c)
    {
      mCoefficients[0] = a;
      mCoefficients[1] = b;
      mCoefficients[2] = c;
    } // end Quadratic::setCoefficients()

    /*! This method returns the squared term coefficient.
     *  \return mCoefficients[0].
     */
    inline float a(void) const
    {
      return mCoefficients[0];
    } // end Quadratic::a()

    /*! This method returns the squared term coefficient.
     *  \return mCoefficients[0].
     */
    inline float &a(void)
    {
      return mCoefficients[0];
    } // end Quadratic::a()

    /*! This method returns the linear term coefficient.
     *  \return mCoefficients[1].
     */
    inline float &b(void)
    {
      return mCoefficients[1];
    } // end Quadratic::b()

    /*! This method returns the linear term coefficient.
     *  \return mCoefficients[1].
     */
    inline float b(void) const
    {
      return mCoefficients[1];
    } // end Quadratic::b()

    /*! This method returns the squared term coefficient.
     *  \return mCoefficients[2].
     */
    inline float &c(void)
    {
      return mCoefficients[2];
    } // end Quadratic::c()

    /*! This method returns the squared term coefficient.
     *  \return mCoefficients[2].
     */
    inline float c(void) const
    {
      return mCoefficients[2];
    } // end Quadratic::c()

    /*! This method returns the real roots of this Quadratic, if they exist.
     *  \param x0 The first real root of this Quadratic, if it exists.
     *  \param x1 The second real root of this Quadratic, if it exists.
     *  \return The number of real roots of this Quadratic, if they exist: 0,1, or 2.
     */
    inline int realRoots(float &x0, float &x1) const;

  protected:
    /*! A Quadratic is defined by three coefficients:
     *  a = mCoefficients[0] \n
     *  b = mCoefficients[1] \n
     *  c = mCoefficients[2]
     */
    float mCoefficients[3];
}; // end class Quadratic

#include "Quadratic.inl"

#endif // QUADRATIC_H
