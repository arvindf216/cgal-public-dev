
// Copyright (c) 2018 GeometryFactory (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
// You can redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $URL$
// $Id$
// SPDX-License-Identifier: GPL-3.0+
//
//
// Author(s) : Maxime Gimeno
//

#ifndef CGAL_AABB_DO_INTERSECT_TRANSFORM_TRAITS_H
#define CGAL_AABB_DO_INTERSECT_TRANSFORM_TRAITS_H

#include <CGAL/license/AABB_tree.h>

#include <CGAL/Bbox_3.h>
#include <CGAL/intersections.h>
#include <CGAL/Cartesian_converter.h>
#include <CGAL/internal/AABB_tree/Has_nested_type_Shared_data.h>
#include <CGAL/internal/AABB_tree/Is_ray_intersection_geomtraits.h>
#include <CGAL/internal/AABB_tree/Primitive_helper.h>
#include <CGAL/Filtered_predicate.h>

#include <CGAL/Aff_transformation_3.h>
#include <boost/mpl/if.hpp>

/// \file AABB_do_intersect_transform_traits.h

namespace CGAL {

template<typename Kernel, typename AABBPrimitive>
class AABB_do_intersect_transform_traits:
    public AABB_traits<Kernel, AABBPrimitive>
{
  mutable Aff_transformation_3<Kernel> m_transfo;
  typedef AABB_traits<Kernel, AABBPrimitive> BaseTraits;
  typedef AABB_do_intersect_transform_traits<Kernel, AABBPrimitive> Self;
public:

  //Constructor
  AABB_do_intersect_transform_traits(const Aff_transformation_3<Kernel>& transf = Aff_transformation_3<Kernel>(IDENTITY))
    :m_transfo(transf)
  {}

  // AABBTraits concept types
  typedef typename BaseTraits::Point_3 Point_3;
  typedef typename BaseTraits::Primitive Primitive;
  typedef typename BaseTraits::Bounding_box Bounding_box;

  // helper functions
  Bbox_3
  static compute_transformed_bbox(const Bbox_3& bbox, const Aff_transformation_3<Kernel>& transfo)
  {
    // TODO: possible optimization using Protector
    typedef Simple_cartesian<Interval_nt<> >             Approximate_kernel;
    typedef Cartesian_converter<Kernel, Approximate_kernel>    C2F;
    C2F c2f;

    Approximate_kernel::Aff_transformation_3 af = c2f(transfo);

    //TODO reuse the conversions
    typename Approximate_kernel::Point_3 ps[8];
    ps[0] = af( c2f( Point_3(bbox.min(0), bbox.min(1), bbox.min(2)) ) );
    ps[1] = af( c2f( Point_3(bbox.min(0), bbox.min(1), bbox.max(2)) ) );
    ps[2] = af( c2f( Point_3(bbox.min(0), bbox.max(1), bbox.min(2)) ) );
    ps[3] = af( c2f( Point_3(bbox.min(0), bbox.max(1), bbox.max(2)) ) );

    ps[4] = af( c2f( Point_3(bbox.max(0), bbox.min(1), bbox.min(2)) ) );
    ps[5] = af( c2f( Point_3(bbox.max(0), bbox.min(1), bbox.max(2)) ) );
    ps[6] = af( c2f( Point_3(bbox.max(0), bbox.max(1), bbox.min(2)) ) );
    ps[7] = af( c2f( Point_3(bbox.max(0), bbox.max(1), bbox.max(2)) ) );

    return bbox_3(ps, ps+8);
  }

  Bbox_3
  compute_transformed_bbox(const Bbox_3& bbox) const
  {
    return compute_transformed_bbox(bbox, m_transfo);
  }

  // Do_intersect predicate
  class Do_intersect
    : BaseTraits::Do_intersect
  {
    typedef AABB_do_intersect_transform_traits<Kernel, AABBPrimitive> AABBTraits;
    const AABBTraits& m_traits;
    typedef typename BaseTraits::Do_intersect Base;

    Bounding_box
    compute_transformed_bbox(const Bounding_box& bbox) const
    {
      return m_traits.compute_transformed_bbox(bbox);
    }

  public:
    Do_intersect(const AABBTraits& traits)
    : Base(static_cast<const BaseTraits&>(traits)),
      m_traits(traits)
    {}

    template<typename Query>
    bool operator()(const Query& q, const Bounding_box& bbox) const
    {
      return
        static_cast<const Base*>(this)->operator()(
          q, compute_transformed_bbox(bbox));
    }

    template<typename Query>
    bool operator()(const Query& q, const Primitive& pr) const
    {
      // transformation is done within Primitive_helper
      return do_intersect(q, internal::Primitive_helper<Self>::get_datum(pr,m_traits));
    }

    // intersection with AABB-tree
    template<typename AABBTraits>
    bool operator()(const CGAL::AABB_tree<AABBTraits>& other_tree, const Primitive& pr) const
    {
      // transformation is done within Primitive_helper
      return other_tree.do_intersect( internal::Primitive_helper<Self>::get_datum(pr,m_traits));
    }

    template<typename AABBTraits>
    bool operator()(const CGAL::AABB_tree<AABBTraits>& other_tree, const Bounding_box& bbox) const
    {
      return other_tree.do_intersect(compute_transformed_bbox(bbox));
    }
  };

  Do_intersect do_intersect_object() const{
    return Do_intersect(*this);
  }

  //Specific
  void set_transformation(const Aff_transformation_3<Kernel>& trans) const
  {
    m_transfo = trans;
  }

  const Aff_transformation_3<Kernel>& transformation() const { return m_transfo; }
};

namespace internal {

template<typename K, typename P>
struct Primitive_helper<AABB_do_intersect_transform_traits<K,P> ,true>{

typedef AABB_do_intersect_transform_traits<K,P> Traits;


static typename Traits::Primitive::Datum get_datum(const typename Traits::Primitive& p,
                            const Traits & traits)
{
  return p.datum(traits.shared_data()).transform(traits.transformation());
}

static typename Traits::Point_3 get_reference_point(const typename Traits::Primitive& p,const Traits& traits) {
  return p.reference_point(traits.shared_data()).transform(traits.transformation());
}

};

template<typename K, typename P>
typename CGAL::AABB_tree<AABB_do_intersect_transform_traits<K,P> >::Bounding_box
get_tree_bbox(const CGAL::AABB_tree<AABB_do_intersect_transform_traits<K,P> >& tree)
{
  return tree.traits().compute_transformed_bbox(tree.bbox());
}

template<typename K, typename P>
typename CGAL::AABB_tree<AABB_do_intersect_transform_traits<K,P> >::Bounding_box
get_node_bbox(const CGAL::AABB_node<AABB_do_intersect_transform_traits<K,P> >& node,
              const AABB_do_intersect_transform_traits<K,P>& traits)
{
  return traits.compute_transformed_bbox(node.bbox());
}

} // end internal

}//end CGAL

#endif //CGAL_AABB_AABB_do_intersect_transform_traits_H
