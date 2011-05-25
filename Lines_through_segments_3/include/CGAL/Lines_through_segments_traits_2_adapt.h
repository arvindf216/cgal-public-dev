// Copyright (c) 2010  Tel-Aviv University (Israel).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org); you may redistribute it under
// the terms of the Q Public License version 1.0.
// See the file LICENSE.QPL distributed with CGAL.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $URL: $
// $Id: $
// 
//
// Author(s)     : Asaf Porat          <asafpor1@post.tau.ac.il>

#ifndef LINES_THROUGH_SEGMENTS_TRAITS_2_ADAPT_H
#define LINES_THROUGH_SEGMENTS_TRAITS_2_ADAPT_H

/*! \file
 *
 * The file contains different classes to adapt between the different 
 * arrangement traits classes.
 * Currently, Conic and algebraic traits classes are supported.
 */

#include <CGAL/Algebraic_structure_traits.h>
#include <CGAL/Lines_through_segments_traits_3.h>
#include <CGAL/predicates_on_points_2.h>
#include <CGAL/Lines_through_segments_general_functions.h>

#include <CGAL/Arr_conic_traits_2.h>
#include <CGAL/CORE_algebraic_number_traits.h>
#include <CGAL/Arr_rational_arc_traits_d_1.h>
#include <CGAL/Arr_vertical_segment_traits.h>
#include <CGAL/Algebraic_kernel_d_1.h>

namespace CGAL {
   
template <typename Traits_3_>
class Lines_through_segments_traits_on_plane_adapt
{
public:
   typedef Traits_3_                                     Traits_3;

private:
  typedef typename Traits_3::Rational_kernel            Rational_kernel;
  typedef typename Traits_3::Alg_kernel                 Alg_kernel;
  typedef CORE::BigInt                                  Integer;
      
  typedef typename Traits_3::Traits_arr_on_plane_2      Traits_arr_on_plane_2;
  typedef typename Rational_kernel::Segment_2           Rational_segment_2;
  typedef typename Rational_kernel::Segment_3           Rational_segment_3;
  typedef typename Rational_kernel::Point_2             Rational_point_2;
         
  // typedef typename Traits_arr_on_plane_2::Point_2 Point_2;

  /* Specific typedefs for conic arc traits. */
  typedef CGAL::CORE_algebraic_number_traits            Nt_traits;
  typedef CGAL::Arr_conic_traits_2<Rational_kernel, Alg_kernel, Nt_traits> 
    Conic_traits_arr_on_plane_2;
  typedef typename Conic_traits_arr_on_plane_2::Point_2 Conic_point_2;
  typedef typename Conic_traits_arr_on_plane_2::Curve_2 Conic_curve_2;
  typedef typename Conic_traits_arr_on_plane_2::X_monotone_curve_2
    Conic_x_monotone_curve_2;
  
  /* Specific typedefs for Rational arc traits . */
  typedef CGAL::Arr_rational_arc_traits_d_1<Rational_kernel>
                                                        Traits_d_1;
  typedef CGAL::Arr_traits_with_vertical_segments <Traits_d_1>
    Rational_arc_traits_arr_on_plane_2;
  typedef typename Rational_arc_traits_arr_on_plane_2::Point_2
                                                        Rational_arc_point_2;
  typedef typename Rational_arc_traits_arr_on_plane_2::X_monotone_curve_2
    Rational_arc_x_monotone_curve_2;
  typedef typename Rational_arc_traits_arr_on_plane_2::Curve_2
                                                        Rational_arc_curve_2;

  typedef CORE::Polynomial<Integer>                     Polynomial;
  typedef typename CORE::BigFloat                       CORE_big_float;
  typedef typename CORE::BFInterval                     CORE_big_float_interval;
      
  typedef typename Traits_d_1::Algebraic_kernel         Traits_algebraic_kernel;
  typedef typename Traits_d_1::Polynomial               Polynomial_1;
  typedef typename Traits_algebraic_kernel::Bound       Bound;
  typedef std::pair<Bound,Bound>                        Bound_pair;

public:
  typedef typename Traits_d_1::Algebraic_real_1         Algebraic_real_1;
  typedef typename Traits_3::Algebraic_NT               Algebraic;
  typedef typename Traits_3::Rational_NT                Rational;

  /**************************************************************
   * The following function return the orientation of a curve.
   *
   * Output:
   *      CGAL::COLLINEAR or CGAL::COUNTERCLOCKWISE or CGAL::CLOCKWISE
   ***************************************************************/

  CGAL::Orientation orientation(const Conic_curve_2& curve,
                                const Alg_kernel* ker)
  {
    return this->orientation_conic_traits(curve,ker);
  }

  CGAL::Orientation orientation(const Conic_x_monotone_curve_2& curve,
                                const Alg_kernel* ker)
  {
    return this->orientation_conic_traits(curve,ker);
  }
      

  template <typename Curve_2>
  CGAL::Orientation orientation_conic_traits(const Curve_2& curve,
                                             const Alg_kernel* ker)
  {
    return curve.orientation();
  }
      
  CGAL::Orientation orientation(const Rational_arc_x_monotone_curve_2& curve,
                                const Alg_kernel* ker)
  {
    return this->orientation_rational_arc_traits(curve,ker);
  }
      
  CGAL::Orientation orientation(const Rational_arc_curve_2& curve,
                                const Alg_kernel* ker)
  {
    return this->orientation_rational_arc_traits(curve,ker);
  }
      
  template <typename Curve_2>
  CGAL::Orientation orientation_rational_arc_traits(const Curve_2& curve,
                                                    const Alg_kernel* ker)
  {
    Algebraic left_x;
    Algebraic right_x;
    Algebraic left_y;
    Algebraic right_y;
         
    convert_point (curve.left(),left_x, left_y); 
    convert_point (curve.right(),right_x, right_y); 
         
    Algebraic mid_x = left_x + (right_x - left_x)/2;
    Algebraic mid_y = left_y + (right_y - left_y)/2;

    typename Alg_kernel::Point_2 mid_p(mid_x, mid_y);
    typename Alg_kernel::Point_2 right_p(right_x, right_y);
    typename Alg_kernel::Point_ left_p(left_x, left_y);
    CGAL::Orientation orient = ker->orientation_2_object()(left_p, mid_p,
                                                           right_p);
    if (orient == LEFT_TURN)
    {
      return CGAL::COUNTERCLOCKWISE;
    }
    else if (orient == RIGHT_TURN)
    {
      return CGAL::CLOCKWISE;
    }
    return CGAL::COLLINEAR;
  }

  /**************************************************************
   * The following functions returns the middle point of a curve.
   *
   * Output:
   *      Point_2
   ***************************************************************/

  template <typename Point_2>
  void get_mid_point(const Conic_curve_2& curve, Point_2& output_p)
  {
    return get_mid_point_conic_traits(curve,output_p);
  }
      
  template <typename Point_2>
  void get_mid_point(const Conic_x_monotone_curve_2& curve,
                     Point_2& output_p)
  {
    return get_mid_point_conic_traits(curve,output_p);
  }
      
  template <typename Curve_2,typename Point_2>
  void get_mid_point_conic_traits(const Curve_2& curve, Point_2& output_p)
  {
    Point_2 source = curve.source();
    Point_2 target = curve.target();
         
    Algebraic mid_x = source.x() +
      (target.x()- source.x())/2;
         
    Rational t = curve.t();
    Rational u = curve.u();
    Rational v = curve.v();
    Rational w = curve.w();
         
    if (t == 0 && v == 0)
    {
      output_p = Point_2(mid_x, ((-u) * mid_x - w));
    }
    else
    {
      output_p = Point_2(mid_x, ((-u) * mid_x - w)/(t * mid_x + v));
    }
  }


  template <typename Point_2>
  void get_mid_point(const Rational_arc_curve_2& curve, Point_2& output_p)
  {
    return get_mid_point_rational_arc_traits(curve,output_p);
  }
      

  template <typename Point_2>
  void get_mid_point(const Rational_arc_x_monotone_curve_2& curve,
                     Point_2& output_p)
  {
    return get_mid_point_rational_arc_traits(curve,output_p);
  }
      
  template <typename Curve_2,typename Point_2>
  void get_mid_point_rational_arc_traits(const Curve_2& curve,
                                         Point_2& output_p)
  {

    Point_2 source = curve.source();
    Point_2 target = curve.target();
         
    Algebraic mid_x = source.x() +
      (target.x()- source.x())/2;

    Polynomial core_numer = convert_polynomial(curve.numerator());
    Polynomial core_denom = convert_polynomial(curve.denominator());

    Algebraic y_numer = core_numer.eval(mid_x);
    Algebraic y_denom = core_denom.eval(mid_x);

    Algebraic mid_y(y_numer / y_denom);

    output_p = Point_2(mid_x,mid_y);
  }

  /**************************************************************
   * The following functions returns the y val, given a x val on curve.
   *
   * Output:
   *      Point_2
   ***************************************************************/

  template <typename Number_type>
  Algebraic get_y_val(const Conic_curve_2& curve, const Number_type& x)
  {
    return get_y_val_conic_traits(curve,x);
  }
      
  template <typename Number_type>
  Algebraic get_y_val(const Conic_x_monotone_curve_2& curve,
                      const Number_type& x)
  {
    return get_y_val_conic_traits(curve,x);
  }
      
  template <typename Curve_2,typename Number_type>
  Algebraic get_y_val_conic_traits(const Curve_2& curve, const Number_type& x)
  {
    Rational t = curve.t();
    Rational u = curve.u();
    Rational v = curve.v();
    Rational w = curve.w();

    /* Get the y value of curve at x */
    Algebraic temp_y;
    if (t == 0 && v == 0)
    {
      temp_y = ((-u) * x - w);
    }
    else
    {
      temp_y = (((-u) * x - w)/(t * x + v));
    }
         
    return temp_y;
  }


  template <typename Number_type>
  Algebraic get_y_val(const Rational_arc_curve_2& curve,
                      const Number_type& x)
  {
    return get_y_val_rational_arc_traits(curve,x);
  }

  Rational get_x_val(const Rational_arc_x_monotone_curve_2& curve,
                     const Rational& y)
  {
    return get_x_val_rational_arc_traits(curve,y);
  }
            
  Rational get_x_val(const Rational_arc_curve_2& curve,
                     const Rational& y)
  {
    return get_x_val_rational_arc_traits(curve,y);
  }

  template <typename Number_type>
  Algebraic get_y_val(const Rational_arc_x_monotone_curve_2& curve,
                      const Number_type& x)
  {
    return get_y_val_rational_arc_traits(curve,x);
  }
      
  template <typename Curve_2>
  Algebraic get_y_val_rational_arc_traits(const Curve_2& curve,
                                          const Algebraic_real_1& _x)
  {
    Algebraic x = convert_real_to_algebraic(_x);
    return get_y_val_rational_arc_traits(curve,x);
  }

  template <typename Curve_2>
  Algebraic get_y_val_rational_arc_traits(const Curve_2& curve,
                                          const Algebraic& x)
  {
    Polynomial core_numer = convert_polynomial(curve.numerator());
    Polynomial core_denom = convert_polynomial(curve.denominator());

    Algebraic y_numer = core_numer.eval(x);
    Algebraic y_denom = core_denom.eval(x);

    Algebraic y(y_numer / y_denom);
    return y;
  }

  template <typename Curve_2>
  Rational get_x_val_rational_arc_traits(const Curve_2& curve,
                                         const Rational& y)
  {
    Polynomial core_numer = convert_polynomial(curve.numerator());
    Polynomial core_denom = convert_polynomial(curve.denominator());
    Integer x_0_num = core_numer.getCoeffi(0);
    Integer x_1_num = core_numer.getCoeffi(1);
    Integer x_0_denom = core_denom.getCoeffi(0);
    Integer x_1_denom = core_denom.getCoeffi(1);

    Rational x = (y * x_0_denom - x_0_num) / (x_1_num - x_1_denom * y);
         
    return x;
  }
  template <typename Curve_2>
  void get_horizontal_asymptote_y_val(const Curve_2& curve,
                                      Algebraic& y)
  {
    Polynomial core_numer = convert_polynomial(curve.numerator());
    Polynomial core_denom = convert_polynomial(curve.denominator());
    CGAL_assertion(CGAL::degree(curve.numerator()) == 
                   CGAL::degree(curve.denominator()));
         
    if (CGAL::degree(curve.numerator()) == 1)
    {
      Integer x_1_num = core_numer.getCoeffi(1);
      Integer x_1_denom = core_denom.getCoeffi(1);
      y = Algebraic(x_1_num) / Algebraic(x_1_denom);
    }
    else //if (core_numer.degree() == 0)
    {
      CGAL_assertion(CGAL::degree(curve.denominator()) == 0);
      Integer x_0_num = core_numer.getCoeffi(0);
      Integer x_0_denom = core_denom.getCoeffi(0);
      y = Algebraic(x_0_num) / Algebraic(x_0_denom);
    }
  }

  /**************************************************************
   * The following function adapts creation of rational segment
   * on plane arr.
   *
   * Input:
   *      source - Segment_2 source point.
   *      target  - Segment_2 end point.
   *
   ***************************************************************/

  void create_segment_on_plane_arr(Conic_curve_2& cv, 
                                   const Rational_point_2& source,
                                   const Rational_point_2& target)
  {
    cv = Conic_curve_2(Rational_segment_2(source, target));
  }

  void create_segment_on_plane_arr(Rational_arc_curve_2& cv, 
                                   const Rational_arc_point_2& source,
                                   const Rational_arc_point_2& target)
  {
    cv = Rational_arc_curve_2(source, target);
  }

  void create_segment_on_plane_arr(Rational_arc_curve_2& cv, 
                                   const Rational_point_2& source,
                                   const Rational_point_2& target)
  {
    std::vector<Rational>        P (2);

    /* Create vertical segment */
    if (source.x() == target.x())
    {
      Rational_arc_point_2 ps(source.x(),source.y());
      Rational_arc_point_2 pt(target.x(),target.y());
      cv = Rational_arc_curve_2(ps, pt);
    }
    else
    {
      Algebraic_real_1 xs(source.x());
      Algebraic_real_1 xt(target.x());
            
      P[1] = (target.y() - source.y())/(target.x() - source.x());
      P[0] = source.y() - P[1] * source.x();
            
      // std::cout << "Segment = (" << xs << "," << xt << ")   " << 
      //    P[0] << " + x * " << P[1] << std::endl;
      cv = Rational_arc_curve_2(P, xs, xt);
    }
  }

  /**************************************************************
   * The following function adapts creation of rational curve_2
   * on plane arr.
   * The curve is of the type:
   *               
   *            V[3]*xy + V[2]*x + v[1]*y + V[0] = 0
   *
   *            y = (-v[0] -v[2]*x)/(v[1] + v[3]*x)
   *
   * Input:
   *      source - Curve source point.
   *      target - Curve end point.
   *      V[]    - Curve coefficients.
   *
   *
   ***************************************************************/

  void create_curve_on_plane_arr(Conic_curve_2& cv,
                                 const Rational& source_x,
                                 const Rational& target_x,
                                 const Rational coefficients[4])
         
  {
    CGAL_assertion((target_x * coefficients[3] + coefficients[1]) != 
                   Rational(0));
    CGAL_assertion((source_x * coefficients[3] + coefficients[1]) != 
                   Rational(0));
         
    Rational_point_2 source(source_x,
                            (source_x * (- coefficients[2]) - coefficients[0])/
                            (source_x * coefficients[3] + coefficients[1]));

    Rational_point_2 target(target_x,
                            (target_x * (- coefficients[2]) - coefficients[0])/
                            (target_x * coefficients[3] + coefficients[1]));

    Conic_point_2 a_source(source.x(), source.y());
    Conic_point_2 a_target(target.x(), target.y());
        
    create_curve_on_plane_arr(cv, a_source, a_target, coefficients);
  }
         
private:
  void create_curve_on_plane_arr(Conic_curve_2& cv,
                                 const Conic_point_2& a_source,
                                 const Conic_point_2& a_target,
                                 const Rational coefficients[4])
         
  {      
    Algebraic mid_point_y = (a_target.y() + a_source.y())/2;
    Algebraic mid_point_x = (a_target.x() + a_source.x())/2;
    Algebraic mid_point_on_hyp_y =
      ((mid_point_x * (-coefficients[2]) + (-coefficients[0])) /
       (mid_point_x * coefficients[3] + coefficients[1]));
         
    if (mid_point_y < mid_point_on_hyp_y)
    {
      cv = Conic_curve_2(Rational(0), Rational(0),
                         coefficients[3], coefficients[2],
                         coefficients[1], coefficients[0],
                         CGAL::CLOCKWISE, a_source, a_target);
    }
    else if (mid_point_y > mid_point_on_hyp_y)
    {
      cv = Conic_curve_2(Rational(0), Rational(0),
                         coefficients[3], coefficients[2],
                         coefficients[1], coefficients[0],
                         CGAL::COUNTERCLOCKWISE, a_source, a_target);
    }
    else
    {
      cv = Conic_curve_2(Rational(0), Rational(0),
                         coefficients[3], coefficients[2],
                         coefficients[1], coefficients[0],
                         CGAL::COLLINEAR, a_source, a_target);
    }
  }
      
public:
  void create_curve_on_plane_arr(Rational_arc_curve_2& cv,
                                 const Rational& source_x,
                                 const Rational& target_x,
                                 const Rational coefficients[4])
  {
    std::vector<Rational>        P2(2);
    P2[0] = -coefficients[0];
    P2[1] = -coefficients[2];
         
    std::vector<Rational>        Q2(2);
    Q2[0] = coefficients[1];
    Q2[1] = coefficients[3];
         
    Algebraic_real_1 xs(source_x);
    Algebraic_real_1 xt(target_x);
    cv = Rational_arc_curve_2(P2, Q2,xs,xt);
  }
  
  template <typename NT1, typename NT2>
  void create_curve_on_plane_arr(Rational_arc_curve_2& new_cv,
                                 const NT1& source_x,
                                 const NT2& target_x,
                                 const Rational_arc_curve_2& old_cv)
  {
    create_curve_on_plane_arr_pr(new_cv,
                                 source_x,
                                 target_x,
                                 old_cv);
  }

  template <typename NT1,typename NT2>
  void create_curve_on_plane_arr(Rational_arc_curve_2& new_cv,
                                 const NT1& source_x,
                                 const NT2& target_x,
                                 const Rational_arc_x_monotone_curve_2&old_cv)
  {
    create_curve_on_plane_arr_pr(new_cv,
                                 source_x,
                                 target_x,
                                 old_cv);
  }
         
  template <typename NT1,typename NT2, typename Curve_on_arr_2>
  /* Curve on arr may be either x monotone or not x monotoe curve. */     
  void create_curve_on_plane_arr_pr(Rational_arc_curve_2& new_cv,
                                    const NT1& source_x,
                                    const NT2& target_x,
                                    const Curve_on_arr_2& old_cv)
  {
    Polynomial core_numer = convert_polynomial(old_cv.numerator());
    Polynomial core_denom = convert_polynomial(old_cv.denominator());

    std::vector<Rational>        P2(2);
    P2[0] = core_numer.getCoeffi(0);
    P2[1] = core_numer.getCoeffi(1);
       
    std::vector<Rational>        Q2(2);
    Q2[0] = core_denom.getCoeffi(0);
    Q2[1] = core_denom.getCoeffi(1);
       
    Algebraic_real_1 xs(source_x);
    Algebraic_real_1 xt(target_x);
    new_cv = Rational_arc_curve_2(P2, Q2,xs,xt);
  }

  bool is_vertical(const Rational_arc_curve_2& arc)
  {
    return (arc.source_infinite_in_x() == CGAL::ARR_INTERIOR &&
            arc.target_infinite_in_x() == CGAL::ARR_INTERIOR &&
            arc.source_x() == arc.target_x());
  }
         
  void create_curve_on_plane_arr(Rational_arc_curve_2& new_cv,
                                 const Rational_arc_x_monotone_curve_2& old_cv)
  {
    typename Rational_arc_traits_arr_on_plane_2::Is_vertical_2 is_ver_obj;
    if (is_ver_obj(old_cv))
    {
      new_cv = Rational_arc_curve_2(old_cv.source(), old_cv.target());
      return;
    }
       
    Polynomial core_numer = convert_polynomial(old_cv.numerator());
    Polynomial core_denom = convert_polynomial(old_cv.denominator());

    std::vector<Rational>        P2(2);
    P2[0] = core_numer.getCoeffi(0);
    P2[1] = core_numer.getCoeffi(1);
       
    std::vector<Rational>        Q2(2);
    Q2[0] = core_denom.getCoeffi(0);
    Q2[1] = core_denom.getCoeffi(1);
       
    Algebraic_real_1 xs(old_cv.source_x());
    Algebraic_real_1 xt(old_cv.target_x());
    new_cv = Rational_arc_curve_2(P2, Q2, xs, xt);
  }

  void create_curve_on_plane_arr(Rational_arc_curve_2& cv,
                                 const Rational& source_x,
                                 bool dir_right,
                                 const Rational coefficients[4])
  {
    std::vector<Rational>        P2(2);
    P2[0] = -coefficients[0];
    P2[1] = -coefficients[2];
       
    std::vector<Rational>        Q2(2);
    Q2[0] = coefficients[1];
    Q2[1] = coefficients[3];
       
    Algebraic_real_1 xs(source_x);

    cv = Rational_arc_curve_2(P2, Q2,xs,dir_right);
  }

  void create_curve_on_plane_arr(Rational_arc_curve_2& cv,
                                 const Rational coefficients[4])
  {
    std::vector<Rational>        P2(2);
    P2[0] = -coefficients[0];
    P2[1] = -coefficients[2];
       
    std::vector<Rational>        Q2(2);
    Q2[0] = coefficients[1];
    Q2[1] = coefficients[3];

    cv = Rational_arc_curve_2(P2, Q2);
  }

  void create_vertical_segment_on_plane_arr(Rational_arc_curve_2& cv, 
                                            const Rational_point_2& source)
  {
    cv = Rational_arc_curve_2(Rational_arc_point_2(source.x(),source.y()));
  }

  void create_vertical_segment_on_plane_arr(Rational_arc_curve_2& cv,
                                            const Rational_point_2& source,
                                            bool is_dir_up)
  {
    cv = Rational_arc_curve_2(Rational_arc_point_2(source.x(), source.y()),
                              is_dir_up);
  }

  void create_vertical_segment_on_plane_arr(Conic_curve_2& cv, 
                                            const Rational_point_2& source)
  {
    CGAL_error_msg("Conic arc traits do not support infinite vertical line.");
  }

  void create_vertical_segment_on_plane_arr(Conic_curve_2& cv,
                                            const Rational_point_2& source,
                                            bool is_dir_up)
  {
    CGAL_error_msg("Conic arc traits do not support infinite vertical line.");
  }

  void create_horizontal_curve_on_plane_arr(Rational_arc_curve_2& cv,
                                            const Rational& S2_t,
                                            const Rational& source_x,
                                            bool dir_right)
  {
    create_horizontal_curve_on_plane_arr_pr(cv,S2_t,source_x,dir_right);
  }

  void create_horizontal_curve_on_plane_arr_pr(Rational_arc_x_monotone_curve_2& cv,
                                               const Rational& S2_t,
                                               const Rational& source_x,
                                               bool dir_right)
  {
    create_horizontal_curve_on_plane_arr_pr(cv,S2_t,source_x,dir_right);
  }
         
private:
  template <typename Curve_2>
  void create_horizontal_curve_on_plane_arr_pr(Curve_2& cv,
                                               const Rational& S2_t,
                                               const Rational& source_x,
                                               bool dir_right)
  {
    std::vector<Rational> P2(1);
    P2[0] = S2_t;
       
    std::vector<Rational> Q2(1);
    Q2[0] = Rational(1);

    Algebraic_real_1 xs(source_x);

    cv = Rational_arc_curve_2(P2, Q2,xs,dir_right);
  }
public:
  void create_horizontal_curve_on_plane_arr(Rational_arc_curve_2& cv,
                                            const Rational& S2_t)
  {
    create_horizontal_curve_on_plane_arr_pr(cv,S2_t);
  }

  void create_horizontal_curve_on_plane_arr(Rational_arc_x_monotone_curve_2& cv,
                                            const Rational& S2_t)
  {
    create_horizontal_curve_on_plane_arr_pr(cv,S2_t);
  }

private:
  template <typename Curve_2>
  void create_horizontal_curve_on_plane_arr_pr(Curve_2& cv,
                                               const Rational& S2_t)
  {
    std::vector<Rational>        P2(1);
    P2[0] = S2_t;
       
    std::vector<Rational>        Q2(1);
    Q2[0] = Rational(1);

    cv = Rational_arc_curve_2(P2, Q2);
  }
public:
  template <typename NT1, typename NT2>
  void create_horizontal_curve_on_plane_arr(Rational_arc_curve_2& cv,
                                            const Rational& S2_t,
                                            const NT1& source_x,
                                            const NT2& target_x)
  {
    std::vector<Rational>        P2(1);
    P2[0] = S2_t;
       
    std::vector<Rational>        Q2(1);
    Q2[0] = Rational(1);

       
    Algebraic_real_1 xs(source_x);
    Algebraic_real_1 xt(target_x);

    cv = Rational_arc_curve_2(P2, Q2, xs, xt);
  }

  bool is_horizontal(const Rational_arc_curve_2& curve)
  {
    const int    deg_p (CGAL::degree(curve.numerator()));
    const int    deg_q (CGAL::degree(curve.denominator()));

    if (deg_p == 0 && deg_q == 0)
    {
      return true;
    }

    return false;
  }
         
  /**************************************************************
   * The following function adapts creation of rational curve_2
   * on plane arr.
   *
   * Constructs a circular arc going from p1, the source, through p2, p3 
   * and p4 to p5, the target (notice all points have integer coordinates). 
   * Precondition: No three points of the five are not collinear.The curve is
   * of the type:
   *
   * Input:
   *
   *
   ***************************************************************/
public:

  void create_curve_on_plane_arr(Conic_curve_2& cv,
                                 const Rational coefficients[4])
  {
    CGAL_error_msg("Unbounded arcs are not supported with the conic traits");
  }

  void create_curve_on_plane_arr(Conic_curve_2& cv,
                                 const Rational_point_2 point_1,
                                 const Rational_point_2 point_2,
                                 const Rational_point_2 point_3,
                                 const Rational_point_2 point_4,
                                 const Rational_point_2 point_5)
  {
    cv = Conic_curve_2(point_1, point_2, point_3, point_4, point_5);
  }
      
  void create_curve_on_plane_arr(Rational_arc_curve_2& cv,
                                 const Rational_point_2 point_1,
                                 const Rational_point_2 point_2,
                                 const Rational_point_2 point_3,
                                 const Rational_point_2 point_4,
                                 const Rational_point_2 point_5)
  {
    typename Rational_kernel::Conic_2   temp_conic;
    Rational                       coefficients [4];
         
    temp_conic.set (point_1, point_2, point_3, point_4, point_5);

    /* 
     * Get the conic coefficients:
     * rx^2 + sy^2 + txy + ux + vy + w = 0 
     * r and s equal to 0.
     *
     * txy + ux + vy + w = 0 
     */
    CGAL_precondition((temp_conic.r() == 0) && (temp_conic.s() == 0));

    coefficients[3] = temp_conic.t();
    coefficients[2] = temp_conic.u();
    coefficients[1] = temp_conic.v();
    coefficients[0] = temp_conic.w();
         
    create_curve_on_plane_arr(cv, point_1.x(), point_5.x(), coefficients);
  }

  void create_horizontal_curve_on_plane_arr(Conic_curve_2& cv,
                                            const Rational& S2_t,
                                            const Rational& source_x,
                                            bool dir_right)
  {
    CGAL_error_msg("Unbounded arcs are not supported with the conic traits");
  }

  void create_horizontal_curve_on_plane_arr(Conic_curve_2& cv,
                                            const Rational& S2_t)
  {
    CGAL_error_msg("Unbounded arcs are not supported with the conic traits");
  }
      
  void create_horizontal_curve_on_plane_arr(Conic_curve_2& cv,
                                            const Rational& S2_t,
                                            const Rational& source_x,
                                            const Rational& target_x)
  {
    Rational_point_2 p1(source_x, S2_t);
    Rational_point_2 p2(target_x, S2_t);
       
    create_segment_on_plane_arr(cv,p1,p2);
  }

  /**************************************************************
   * The following function adapts creation of Point_2
   * on plane arr.
   *
   * Input:
   *      
   *
   *
   ***************************************************************/
  typename Traits_arr_on_plane_2::Point_2
  construct_point(const Rational& x, const Rational& y)
  {
    typedef typename Traits_arr_on_plane_2::Point_2 Point_2;
    return Point_2(x,y);
  }

private:
  Polynomial convert_polynomial(const Polynomial_1 poly) const
  {
    const int    d = CGAL::degree(poly);
    Integer* coeffs = new Integer[d+1];
    Polynomial core_poly;
    for (int ii = 0 ; ii <= d; ++ii)
    {
      coeffs[ii] = CGAL::get_coefficient(poly,ii);
    }

    core_poly = Polynomial (d,coeffs);
         
    return core_poly;
  }

public:
  Algebraic convert_real_to_algebraic(const Algebraic_real_1& r)  const
  {
    typename Traits_algebraic_kernel::Compute_polynomial_1 compute_poly;
    Polynomial_1 poly = compute_poly(r);

    typename Traits_algebraic_kernel::Isolate_1 isolate;
    Bound_pair bound_pair = isolate(r,poly);
    CORE_big_float_interval bound_interval(bound_pair);

    CORE_big_float f = bound_interval.first;
    CORE_big_float s = bound_interval.second;

    if ((f.isExact() == false) || (s.isExact() == false))
    {
      bound_interval.first.makeExact();
      bound_interval.second.makeExact();

      bound_interval.first  -= 0.000000000001;
      bound_interval.second += 0.000000000001;
    }
    Polynomial core_poly = convert_polynomial (poly);
    return Algebraic (core_poly, bound_interval);
  }

  template <typename Point_2,typename Number_type>
  void convert_point (const Point_2& p,Number_type& x, Number_type& y) const
  {
    x = convert_real_to_algebraic(p.x());
    Polynomial core_numer = convert_polynomial(p.rational_function().numer());
    Polynomial core_denom = convert_polynomial(p.rational_function().denom());

    Algebraic y_numer = core_numer.eval(x);
    Algebraic y_denom = core_denom.eval(x);

    y = (y_numer / y_denom);
  } 
};

template <typename Traits_3_>
class Lines_through_segments_get_algebraic_number_adapt
{
private:
  typedef Traits_3_                                     Traits_3;

  typedef typename Traits_3::Algebraic_NT               Algebraic;
  typedef CORE::BigInt                                  Integer;
  typedef typename Traits_3::Rational_kernel            Rational_kernel;
  typedef CGAL::Arr_rational_arc_traits_d_1<Rational_kernel>
                                                        Traits_d_1;
  typedef CGAL::Arr_traits_with_vertical_segments <Traits_d_1> 
    Rational_arc_traits_arr_on_plane_2;
  typedef typename Traits_d_1::Algebraic_real_1         Algebraic_real_1;

private:
  template <typename NT>
  Algebraic get_algebraic_number(const NT &x)
  {
    return Algebraic(x);
  }

  Algebraic get_algebraic_number(const Algebraic_real_1 &x)
  {
    Lines_through_segments_traits_on_plane_adapt<Traits_3>
      traits_on_plane_adapt;
    return traits_on_plane_adapt.convert_real_to_algebraic(x);
  }

public:
  template <typename NT>
  Algebraic operator()(const NT &x)
  {
    return get_algebraic_number(x);
  }
};

} //namespace CGAL

#endif /*LINES_THROUGH_SEGMENTS_TRAITS_2_ADAPT_H*/
