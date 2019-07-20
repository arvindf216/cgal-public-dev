// Copyright (c) 2015 GeometryFactory (France).
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
// Author(s)     : Maxime Gimeno, Sebastien Loriot, Martin Skrodzki

#ifndef CGAL_POLYGON_MESH_PROCESSING_DISTANCE_H
#define CGAL_POLYGON_MESH_PROCESSING_DISTANCE_H

#include <CGAL/license/Polygon_mesh_processing/distance.h>

#include <algorithm>
#include <cmath>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/utility.h>
#include <CGAL/Polygon_mesh_processing/internal/named_function_params.h>
#include <CGAL/Polygon_mesh_processing/internal/named_params_helper.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>

#include <CGAL/spatial_sort.h>
#include <CGAL/Polygon_mesh_processing/measure.h>

#include <CGAL/Polygon_mesh_processing/internal/mesh_to_point_set_hausdorff_distance.h>
#include <CGAL/Polygon_mesh_processing/internal/AABB_traversal_traits_with_Hausdorff_distance.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/atomic.h>
#endif // CGAL_LINKED_WITH_TBB

#include <boost/unordered_set.hpp>

#include <limits>

namespace CGAL{
namespace Polygon_mesh_processing {
namespace internal{
template <class Kernel, class OutputIterator>
OutputIterator
triangle_grid_sampling( const typename Kernel::Point_3& p0,
                        const typename Kernel::Point_3& p1,
                        const typename Kernel::Point_3& p2,
                        double distance,
                        OutputIterator out)
{
  typename Kernel::Compute_squared_distance_3 squared_distance;
  const double d_p0p1 = to_double(approximate_sqrt( squared_distance(p0, p1) ));
  const double d_p0p2 = to_double(approximate_sqrt( squared_distance(p0, p2) ));

  const double n = (std::max)(std::ceil( d_p0p1 / distance ),
                              std::ceil( d_p0p2 / distance ));

  for (double i=1; i<n; ++i)
    for (double j=1; j<n-i; ++j)
    {
      const double c0=(1-(i+j)/n), c1=i/n, c2=j/n;
      *out++=typename Kernel::Point_3(
              p0.x()*c0+p1.x()*c1+p2.x()*c2,
              p0.y()*c0+p1.y()*c1+p2.y()*c2,
              p0.z()*c0+p1.z()*c1+p2.z()*c2
            );
    }
  return out;
}

#if defined(CGAL_LINKED_WITH_TBB)
template <class AABB_tree, class Point_3>
struct Distance_computation{
  const AABB_tree& tree;
  const std::vector<Point_3>& sample_points;
  Point_3 initial_hint;
  tbb::atomic<double>* distance;

  Distance_computation(
          const AABB_tree& tree,
          const Point_3& p,
          const std::vector<Point_3>& sample_points,
          tbb::atomic<double>* d)
    : tree(tree)
    , sample_points(sample_points)
    , initial_hint(p)
    , distance(d)
  {}

  void
  operator()(const tbb::blocked_range<std::size_t>& range) const
  {
    Point_3 hint = initial_hint;
    double hdist = 0;
    for( std::size_t i = range.begin(); i != range.end(); ++i)
    {
      hint = tree.closest_point(sample_points[i], hint);
      typename Kernel_traits<Point_3>::Kernel::Compute_squared_distance_3 squared_distance;
      double d = to_double(CGAL::approximate_sqrt( squared_distance(hint,sample_points[i]) ));
      if (d>hdist) hdist=d;
    }

    // update max value stored in distance
    double current_value = *distance;
    while( current_value < hdist )
    {
      current_value = distance->compare_and_swap(hdist, current_value);
    }
  }
};
#endif

template <class Concurrency_tag,
          class Kernel,
          class PointRange,
          class AABBTree>
double approximate_Hausdorff_distance_impl(
  const PointRange& sample_points,
  const AABBTree& tree,
  typename Kernel::Point_3 hint)
{
#if !defined(CGAL_LINKED_WITH_TBB)
  CGAL_static_assertion_msg (!(boost::is_convertible<Concurrency_tag, Parallel_tag>::value),
                             "Parallel_tag is enabled but TBB is unavailable.");
#else
  if (boost::is_convertible<Concurrency_tag,Parallel_tag>::value)
  {
    tbb::atomic<double> distance;
    distance=0;
    Distance_computation<AABBTree, typename Kernel::Point_3> f(tree, hint, sample_points, &distance);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, sample_points.size()), f);
    return distance;
  }
  else
#endif
  {
    double hdist = 0;
    for(const typename Kernel::Point_3& pt : sample_points)
    {
      hint = tree.closest_point(pt, hint);
      typename Kernel::Compute_squared_distance_3 squared_distance;
      typename Kernel::FT dist = squared_distance(hint,pt);
      double d = to_double(CGAL::approximate_sqrt(dist));
      if(d>hdist)
        hdist=d;
    }
    return hdist;
  }
}

} //end of namespace internal

template <class Kernel,
          class FaceRange,
          class TriangleMesh,
          class VertexPointMap,
          class OutputIterator>
OutputIterator
sample_triangles(const FaceRange& triangles,
                 const TriangleMesh& tm,
                 VertexPointMap vpm,
                 double distance,
                 OutputIterator out,
                 bool sample_faces,
                 bool sample_edges,
                 bool add_vertices)
{
  typedef typename boost::property_traits<VertexPointMap>::reference Point_ref;
  typedef typename Kernel::Vector_3 Vector_3;
  typedef boost::graph_traits<TriangleMesh> GT;
  typedef typename GT::face_descriptor face_descriptor;
  typedef typename GT::halfedge_descriptor halfedge_descriptor;

  boost::unordered_set<typename GT::edge_descriptor> sampled_edges;
  boost::unordered_set<typename GT::vertex_descriptor> endpoints;

  for(face_descriptor fd : triangles)
  {
    // sample edges but skip endpoints
    halfedge_descriptor hd = halfedge(fd, tm);
    for (int i=0;i<3; ++i)
    {
      if (sample_edges && sampled_edges.insert(edge(hd, tm)).second )
      {
        Point_ref p0 = get(vpm, source(hd, tm));
        Point_ref p1 = get(vpm, target(hd, tm));
        typename Kernel::Compute_squared_distance_3 squared_distance;
        const double d_p0p1 = to_double(approximate_sqrt( squared_distance(p0, p1) ));

        const double nb_pts = std::ceil( d_p0p1 / distance );
        const Vector_3 step_vec =  typename Kernel::Construct_scaled_vector_3()(
          typename Kernel::Construct_vector_3()(p0, p1),
          typename Kernel::FT(1)/typename Kernel::FT(nb_pts));
        for (double i=1; i<nb_pts; ++i)
        {
          *out++=typename Kernel::Construct_translated_point_3()(p0,
            typename Kernel::Construct_scaled_vector_3()(step_vec ,
              typename Kernel::FT(i)));
        }
      }
      //add endpoints once
      if ( add_vertices && endpoints.insert(target(hd, tm)).second )
        *out++=get(vpm, target(hd, tm));
      hd=next(hd, tm);
    }

    // sample triangles
    if (sample_faces)
    {
      Point_ref p0 = get(vpm, source(hd, tm));
      Point_ref p1 = get(vpm, target(hd, tm));
      Point_ref p2 = get(vpm, target(next(hd, tm), tm));
      out=internal::triangle_grid_sampling<Kernel>(p0, p1, p2, distance, out);
    }
  }
  return out;
}

/** \ingroup PMP_distance_grp
 * generates points taken on `tm` and outputs them to `out`, the sampling method
 * is selected using named parameters.
 * @tparam TriangleMesh a model of the concept `FaceListGraph`
 * @tparam OutputIterator a model of `OutputIterator`
 *  holding objects of the same point type as
 *  the value type of the internal vertex point map of `tm`
 *
 * @param tm the triangle mesh that will be sampled
 * @param out output iterator to be filled with sampled points
 * @param np an optional sequence of \ref pmp_namedparameters "Named Parameters" among the ones listed below
 *
 * \cgalNamedParamsBegin
 *    \cgalParamBegin{vertex_point_map} the property map with the points
 *      associated to the vertices of `tm`. If this parameter is omitted,
 *      an internal property map for `CGAL::vertex_point_t`
 *      must be available for `TriangleMesh`.
 *    \cgalParamEnd
 *    \cgalParamBegin{geom_traits} a model of `PMPDistanceTraits`. \cgalParamEnd
 *    \cgalParamBegin{use_random_uniform_sampling}
 *      if `true` is passed (the default), points are generated in a random
 *      and uniform way on the surface of `tm`, and/or on edges of `tm`.
 *      For faces, the number of sample points is the value passed to the named
 *      parameter `number_of_points_on_faces()`. If not set,
 *      the value passed to the named parameter `number_of_points_per_area_unit()`
 *      is multiplied by the area of `tm` to get the number of sample points.
 *      If none of these parameters is set, the number of points sampled is `num_vertices(tm)`.
 *      For edges, the number of the number of sample points is the value passed to the named
 *      parameter `number_of_points_on_edges()`. If not set,
 *      the value passed to the named parameter `number_of_points_per_distance_unit()`
 *      is multiplied by the sum of the length of edges of `tm` to get the number of sample points.
 *      If none of these parameters is set, the number of points sampled is `num_vertices(tm)`.
 *    \cgalParamEnd
 *    \cgalParamBegin{use_grid_sampling}
 *      if `true` is passed, points are generated on a grid in each triangle,
 *      with a minimum of one point per triangle. The distance between
 *      two consecutive points in the grid is that of the length of the
 *      smallest non-null edge of `tm` or the value passed to
 *      the named parameter `grid_spacing()`. Edges are also split using the
 *      same distance, if requested.
 *    \cgalParamEnd
 *    \cgalParamBegin{use_monte_carlo_sampling}
 *      if `true` is passed, points are generated randomly in each triangle and/or
 *      on each edge.
 *      For faces, the number of points per triangle is the value passed to the named
 *      parameter `number_of_points_per_face()`. If not set, the value passed
 *      to the named parameter `number_of_points_per_area_unit()` is
 *      used to pick a number of points per face proportional to the triangle
 *      area with a minimum of one point per face. If none of these parameters
 *      is set, 2 divided by the square of the length of the smallest non-null
 *      edge of `tm` is used as if it was passed to
 *      `number_of_points_per_area_unit()`.
 *      For edges, the number of points per edge is the value passed to the named
 *      parameter `number_of_points_per_edge()`. If not set, the value passed
 *      to the named parameter `number_of_points_per_distance_unit()` is
 *      used to pick a number of points per edge proportional to the length of
 *      the edge with a minimum of one point per face. If none of these parameters
 *      is set, 1 divided by the length of the smallest non-null edge of `tm`
 *      is used as if it was passed to `number_of_points_per_distance_unit()`.
 *    \cgalParamEnd
 *    \cgalParamBegin{sample_vertices} if `true` is passed (default value),
 *    vertices of `tm` are put into `out`.\cgalParamEnd
 *    \cgalParamBegin{sample_edges} if `true` is passed (default value),
 *    edges of `tm` are sampled.\cgalParamEnd
 *    \cgalParamBegin{sample_faces} if `true` is passed (default value),
 *    faces of `tm` are sampled.\cgalParamEnd
 *    \cgalParamBegin{grid_spacing} a double value used as the grid spacing
 *      for the grid sampling method.
 *    \cgalParamEnd
 *    \cgalParamBegin{number_of_points_on_edges} an unsigned integral value used
 *      for the random sampling method as the number of points to pick exclusively
 *      on edges.
 *    \cgalParamEnd
 *    \cgalParamBegin{number_of_points_on_faces} an unsigned integral value used
 *      for the random sampling method as the number of points to pick on the surface.
 *    \cgalParamEnd
 *    \cgalParamBegin{number_of_points_per_distance_unit} a double value
 *       used for the random sampling and the Monte Carlo sampling methods to
 *       repectively determine the total number of points on edges and the
 *       number of points per edge.
 *    \cgalParamEnd
 *    \cgalParamBegin{number_of_points_per_edge} an unsigned integral value
 *      used by the Monte-Carlo sampling method as the number of points per edge
 *      to pick.
 *    \cgalParamEnd
 *    \cgalParamBegin{number_of_points_per_face} an unsigned integral value
 *      used by the Monte-Carlo sampling method as the number of points per face
 *      to pick.
 *    \cgalParamEnd
 *    \cgalParamBegin{number_of_points_per_area_unit} a double value
 *       used for the random sampling and the Monte Carlo sampling methods to
 *       repectively determine the total number of points inside faces
 *       and the number of points per face.
 *    \cgalParamEnd
 * \cgalNamedParamsEnd
 */
template<class OutputIterator, class TriangleMesh, class NamedParameters>
OutputIterator
sample_triangle_mesh(const TriangleMesh& tm,
                           OutputIterator out,
                           NamedParameters np)
{
  typedef typename GetGeomTraits<TriangleMesh,
          NamedParameters>::type Geom_traits;

  typedef typename GetVertexPointMap<TriangleMesh,
          NamedParameters>::const_type Vpm;

  typedef boost::graph_traits<TriangleMesh> GT;
  typedef typename GT::face_descriptor face_descriptor;
  typedef typename GT::halfedge_descriptor halfedge_descriptor;
  typedef typename GT::edge_descriptor edge_descriptor;

  using boost::choose_param;
  using boost::get_param;
  using boost::is_default_param;

  Vpm pmap = choose_param(get_param(np, internal_np::vertex_point),
                          get_const_property_map(vertex_point, tm));
  typedef Creator_uniform_3<typename Geom_traits::FT,
                            typename Geom_traits::Point_3> Creator;

  Geom_traits geomtraits = choose_param(get_param(np, internal_np::geom_traits), Geom_traits());


  bool use_rs = choose_param(get_param(np, internal_np::random_uniform_sampling), true);
  bool use_gs = choose_param(get_param(np, internal_np::grid_sampling), false);
  bool use_ms = choose_param(get_param(np, internal_np::monte_carlo_sampling), false);

  if (use_gs || use_ms)
    if (is_default_param(get_param(np, internal_np::random_uniform_sampling)))
      use_rs=false;

  bool smpl_vrtcs = choose_param(get_param(np, internal_np::do_sample_vertices), true);
  bool smpl_dgs = choose_param(get_param(np, internal_np::do_sample_edges), true);
  bool smpl_fcs = choose_param(get_param(np, internal_np::do_sample_faces), true);

  double nb_pts_a_u = choose_param(get_param(np, internal_np::nb_points_per_area_unit), 0.);
  double nb_pts_l_u = choose_param(get_param(np, internal_np::nb_points_per_distance_unit), 0.);

  // sample vertices
  if (smpl_vrtcs)
  {
    Property_map_to_unary_function<Vpm> unary(pmap);
    out = std::copy(
      boost::make_transform_iterator(boost::begin(vertices(tm)), unary),
      boost::make_transform_iterator(boost::end(vertices(tm)), unary),
      out);
  }

  // grid sampling
  if (use_gs)
  {
    double grid_spacing_ = choose_param(get_param(np, internal_np::grid_spacing), 0.);
    if (grid_spacing_==0.)
    {
      // set grid spacing to the shortest edge length
      double grid_spacing_ = (std::numeric_limits<double>::max)();
      typedef typename boost::graph_traits<TriangleMesh>
        ::edge_descriptor edge_descriptor;
      for(edge_descriptor ed : edges(tm))
      {
        double el = std::sqrt(
          to_double( typename Geom_traits::Compute_squared_distance_3()(
            get(pmap, source(ed, tm)), get(pmap, target(ed, tm)) )));
        if (el > 0 && el < grid_spacing_)
          grid_spacing_ = el;
      }
    }
    out=sample_triangles<Geom_traits>(
          faces(tm), tm, pmap, grid_spacing_, out,smpl_fcs, smpl_dgs, false);
  }

  // monte carlo sampling
  if (use_ms)
  {
    typename Geom_traits::Compute_squared_distance_3 squared_distance;
    double min_edge_length = (std::numeric_limits<double>::max)();

    std::size_t nb_points_per_face =
      choose_param(get_param(np, internal_np::number_of_points_per_face), 0);
    std::size_t nb_points_per_edge =
      choose_param(get_param(np, internal_np::number_of_points_per_edge), 0);

    if ((nb_points_per_face == 0 && nb_pts_a_u ==0.) ||
        (nb_points_per_edge == 0 && nb_pts_l_u ==0.) )
    {
      typedef typename boost::graph_traits<TriangleMesh>
        ::edge_descriptor edge_descriptor;
      for(edge_descriptor ed : edges(tm))
      {
        double el = std::sqrt(
          to_double( squared_distance(get(pmap, source(ed, tm)),
                                      get(pmap, target(ed, tm)) )));
        if (min_edge_length > 0 && el < min_edge_length)
          min_edge_length = el;
      }
    }

    // sample faces
    if (smpl_fcs)
    {
      // set default value
      if (nb_points_per_face == 0 && nb_pts_a_u ==0.)
        nb_pts_a_u = 2. / CGAL::square(min_edge_length);

      for(face_descriptor f : faces(tm))
      {
        std::size_t nb_points = nb_points_per_face;
        if (nb_points == 0)
        {
          nb_points = (std::max)(
            static_cast<std::size_t>(
              std::ceil(to_double(
                face_area(f,tm,parameters::geom_traits(geomtraits)))*nb_pts_a_u))
          ,std::size_t(1));
        }
        // extract triangle face points
        typename Geom_traits::Point_3 points[3];
        halfedge_descriptor hd(halfedge(f,tm));
        for(int i=0; i<3; ++i)
        {
          points[i] = get(pmap, target(hd, tm));
          hd = next(hd, tm);
        }
        // sample the triangle face
        Random_points_in_triangle_3<typename Geom_traits::Point_3, Creator>
          g(points[0], points[1], points[2]);
        out=std::copy_n(g, nb_points, out);
      }
    }
    // sample edges
    if (smpl_dgs)
    {
      if (nb_points_per_edge == 0 && nb_pts_l_u == 0)
        nb_pts_l_u = 1. / min_edge_length;
      for(edge_descriptor ed : edges(tm))
      {
        std::size_t nb_points = nb_points_per_edge;
        if (nb_points == 0)
        {
          nb_points = (std::max)(
            static_cast<std::size_t>( std::ceil( std::sqrt( to_double(
             squared_distance(get(pmap, source(ed, tm)),
                              get(pmap, target(ed, tm)) )) )*nb_pts_l_u ) ),
            std::size_t(1));
        }
        // now do the sampling of the edge
        Random_points_on_segment_3<typename Geom_traits::Point_3, Creator>
          g(get(pmap, source(ed,tm)), get(pmap, target(ed,tm)));
        out=std::copy_n(g, nb_points, out);
      }
    }
  }

  // random uniform sampling
  if (use_rs)
  {
    // sample faces
    if(smpl_fcs)
    {
      std::size_t nb_points = choose_param(get_param(np, internal_np::number_of_points_on_faces), 0);
      Random_points_in_triangle_mesh_3<TriangleMesh, Vpm, Creator> g(tm, pmap);
      if (nb_points == 0)
      {
        if (nb_pts_a_u == 0.)
          nb_points = num_vertices(tm);
        else
          nb_points = static_cast<std::size_t>(
            std::ceil(g.mesh_area()*nb_pts_a_u) );
      }
      out = std::copy_n(g, nb_points, out);
    }
    // sample edges
    if (smpl_dgs)
    {
      std::size_t nb_points =
        choose_param(get_param(np, internal_np::number_of_points_on_edges), 0);
      Random_points_on_edge_list_graph_3<TriangleMesh, Vpm, Creator> g(tm, pmap);
      if (nb_points == 0)
      {
        if (nb_pts_l_u == 0)
          nb_points = num_vertices(tm);
        else
          nb_points = static_cast<std::size_t>(
            std::ceil( g.mesh_length()*nb_pts_a_u) );
      }
      out = std::copy_n(g, nb_points, out);
    }
  }

  return out;
}

template<class OutputIterator, class TriangleMesh>
OutputIterator
sample_triangle_mesh(const TriangleMesh& tm,
                           OutputIterator out)
{
  return sample_triangle_mesh(tm, out, parameters::all_default());
}

template <class Concurrency_tag,
          class Kernel,
          class PointRange,
          class TriangleMesh,
          class VertexPointMap>
double approximate_Hausdorff_distance(
  const PointRange& original_sample_points,
  const TriangleMesh& tm,
  VertexPointMap vpm)
{
  CGAL_assertion_code(  bool is_triangle = is_triangle_mesh(tm) );
  CGAL_assertion_msg (is_triangle,
        "Mesh is not triangulated. Distance computing impossible.");
  #ifdef CGAL_HAUSDORFF_DEBUG
  std::cout << "Nb sample points " << sample_points.size() << "\n";
  #endif
  typedef typename Kernel::Point_3 Point_3;
  std::vector<Point_3> sample_points
    (boost::begin(original_sample_points), boost::end(original_sample_points) );

  spatial_sort(sample_points.begin(), sample_points.end());

  typedef AABB_face_graph_triangle_primitive<TriangleMesh> Primitive;
  typedef AABB_tree< AABB_traits<Kernel, Primitive> > Tree;

  Tree tree( faces(tm).first, faces(tm).second, tm);
  tree.build();
  tree.accelerate_distance_queries();
  Point_3 hint = get(vpm, *vertices(tm).first);

  return internal::approximate_Hausdorff_distance_impl<Concurrency_tag, Kernel>
    (original_sample_points, tree, hint);
}

template <class Concurrency_tag, class Kernel, class TriangleMesh,
          class NamedParameters,
          class VertexPointMap >
double approximate_Hausdorff_distance(
   const TriangleMesh& tm1,
   const TriangleMesh& tm2,
   NamedParameters np,
   VertexPointMap vpm_2)
{
    std::vector<typename Kernel::Point_3> sample_points;
    sample_triangle_mesh(
                tm1,
                std::back_inserter(sample_points),
                np);
    return approximate_Hausdorff_distance<Concurrency_tag, Kernel>(sample_points, tm2, vpm_2);
}

// documented functions

/**
 * \ingroup PMP_distance_grp
 * computes the approximate Hausdorff distance from `tm1` to `tm2` by returning
 * the distance of the farthest point from `tm2` amongst a sampling of `tm1`
 * generated with the function `sample_triangle_mesh()` with
 * `tm1` and `np1` as parameter.
 *
 * A parallel version is provided and requires the executable to be
 * linked against the <a href="https://www.threadingbuildingblocks.org">Intel TBB library</a>.
 * To control the number of threads used, the user may use the `tbb::task_scheduler_init` class.
 * See the <a href="https://www.threadingbuildingblocks.org/documentation">TBB documentation</a>
 * for more details.
 *
 * @tparam Concurrency_tag enables sequential versus parallel algorithm.
 *                         Possible values are `Sequential_tag`
 *                         and `Parallel_tag`.
 * @tparam TriangleMesh a model of the concept `FaceListGraph`
 * @tparam NamedParameters1 a sequence of \ref pmp_namedparameters "Named Parameters" for `tm1`
 * @tparam NamedParameters2 a sequence of \ref pmp_namedparameters "Named Parameters" for `tm2`
 *
 * @param tm1 the triangle mesh that will be sampled
 * @param tm2 the triangle mesh to compute the distance to
 * @param np1 optional sequence of \ref pmp_namedparameters "Named Parameters" for `tm1` passed to `sample_triangle_mesh()`.
 *
 * @param np2 optional sequence of \ref pmp_namedparameters "Named Parameters" for `tm2` among the ones listed below
 *
 * \cgalNamedParamsBegin
 *    \cgalParamBegin{vertex_point_map} the property map with the points associated to the vertices of `tm2`
 *      If this parameter is omitted, an internal property map for `CGAL::vertex_point_t` must be available in `TriangleMesh`
 *      and in all places where `vertex_point_map` is used.
 *    \cgalParamEnd
 * \cgalNamedParamsEnd
 * The function `CGAL::parameters::all_default()` can be used to indicate to use the default values for
 * `np1` and specify custom values for `np2`
 */
template< class Concurrency_tag,
          class TriangleMesh,
          class NamedParameters1,
          class NamedParameters2>
double approximate_Hausdorff_distance( const TriangleMesh& tm1,
                                       const TriangleMesh& tm2,
                                       const NamedParameters1& np1,
                                       const NamedParameters2& np2)
{
  typedef typename GetGeomTraits<TriangleMesh,
                                 NamedParameters1>::type Geom_traits;

  return approximate_Hausdorff_distance<Concurrency_tag, Geom_traits>(
    tm1, tm2, np1, choose_param(get_param(np2, internal_np::vertex_point),
                                get_const_property_map(vertex_point, tm2)));
}

/**
 * \ingroup PMP_distance_grp
 * computes the approximate symmetric Hausdorff distance between `tm1` and `tm2`.
 * It returns the maximum of `approximate_Hausdorff_distance(tm1, tm2, np1, np2)`
 * and `approximate_Hausdorff_distance(tm2, tm1, np2, np1)`.
 */
template< class Concurrency_tag,
          class TriangleMesh,
          class NamedParameters1,
          class NamedParameters2>
double approximate_symmetric_Hausdorff_distance(
  const TriangleMesh& tm1,
  const TriangleMesh& tm2,
  const NamedParameters1& np1,
  const NamedParameters2& np2)
{
  return (std::max)(
    approximate_Hausdorff_distance<Concurrency_tag>(tm1,tm2,np1,np2),
    approximate_Hausdorff_distance<Concurrency_tag>(tm2,tm1,np2,np1)
  );
}

/**
 * \ingroup PMP_distance_grp
 * returns the distance to `tm` of the point from `points`
 * that is the furthest from `tm`.
 * @tparam PointRange a range of `Point_3`, model of `Range`.
 * @tparam TriangleMesh a model of the concept `FaceListGraph`
 * @tparam NamedParameters a sequence of \ref pmp_namedparameters "Named Parameters"
 * @param points the range of points of interest
 * @param tm the triangle mesh to compute the distance to
 * @param np an optional sequence of \ref pmp_namedparameters "Named Parameters" among the ones listed below
 *
 * \cgalNamedParamsBegin
 *    \cgalParamBegin{vertex_point_map}
 *    the property map with the points associated to the vertices of `tm`. If this parameter is omitted,
 *    an internal property map for `CGAL::vertex_point_t` must be available for the
      vertices of `tm` \cgalParamEnd
 *    \cgalParamBegin{geom_traits} an instance of a geometric traits class, model of `PMPDistanceTraits`\cgalParamEnd
 * \cgalNamedParamsEnd
 */
template< class Concurrency_tag,
          class TriangleMesh,
          class PointRange,
          class NamedParameters>
double max_distance_to_triangle_mesh(const PointRange& points,
                                     const TriangleMesh& tm,
                                     const NamedParameters& np)
{
  typedef typename GetGeomTraits<TriangleMesh,
                                 NamedParameters>::type Geom_traits;

  return approximate_Hausdorff_distance<Concurrency_tag, Geom_traits>
     (points,tm,choose_param(get_param(np, internal_np::vertex_point),
                             get_const_property_map(vertex_point, tm)));
}

/*!
 *\ingroup PMP_distance_grp
 * returns an approximation of the distance between `points` and the point lying on `tm` that is the farthest from `points`
 * @tparam PointRange a range of `Point_3`, model of `Range`.
 * @tparam TriangleMesh a model of the concept `FaceListGraph`
 * @tparam NamedParameters a sequence of \ref pmp_namedparameters "Named Parameters"
 * @param tm a triangle mesh
 * @param points a range of points
 * @param precision for each triangle of `tm`, the distance of its farthest point from `points` is bounded.
 *                  A triangle is subdivided into sub-triangles so that the difference of its distance bounds
 *                  is smaller than `precision`. `precision` must be strictly positive to avoid infinite loops.
 * @param np an optional sequence of \ref pmp_namedparameters "Named Parameters" among the ones listed below
 *
 * \cgalNamedParamsBegin
 *    \cgalParamBegin{vertex_point_map}
 *    the property map with the points associated to the vertices of `tm`. If this parameter is omitted,
 *    an internal property map for `CGAL::vertex_point_t` must be available for the
      vertices of `tm` \cgalParamEnd
 *    \cgalParamBegin{geom_traits} an instance of a geometric traits class, model of `PMPDistanceTraits`. \cgalParamEnd
 * \cgalNamedParamsEnd
 */
template< class TriangleMesh,
          class PointRange,
          class NamedParameters>
double approximate_max_distance_to_point_set(const TriangleMesh& tm,
                                             const PointRange& points,
                                             const double precision,
                                             const NamedParameters& np)
{
  typedef typename GetGeomTraits<TriangleMesh,
                                 NamedParameters>::type Geom_traits;
  typedef boost::graph_traits<TriangleMesh> GT;

  typedef Orthogonal_k_neighbor_search<Search_traits_3<Geom_traits> > Knn;
  typedef typename Knn::Tree Tree;
  Tree tree(points.begin(), points.end());
  CRefiner<Geom_traits> ref;
  for(typename GT::face_descriptor f : faces(tm))
  {
    typename Geom_traits::Point_3 points[3];
    typename GT::halfedge_descriptor hd(halfedge(f,tm));
    for(int i=0; i<3; ++i)
    {
      points[i] = get(choose_param(get_param(np, internal_np::vertex_point),
                                   get_const_property_map(vertex_point, tm)),
                      target(hd, tm));
      hd = next(hd, tm);
    }
    ref.add(points[0], points[1], points[2], tree);
  }
  return to_double(ref.refine(precision, tree));
}

// convenience functions with default parameters

template< class Concurrency_tag,
          class TriangleMesh,
          class PointRange>
double max_distance_to_triangle_mesh(const PointRange& points,
                                     const TriangleMesh& tm)
{
   return max_distance_to_triangle_mesh<Concurrency_tag,
           TriangleMesh,
           PointRange>
           (points, tm, parameters::all_default());
}

template< class TriangleMesh,
          class PointRange>
double approximate_max_distance_to_point_set(const TriangleMesh& tm,
                                             const PointRange& points,
                                             const double precision)
{
  return approximate_max_distance_to_point_set(tm, points, precision,
                                               parameters::all_default());
}

template< class Concurrency_tag,
          class TriangleMesh,
          class NamedParameters>
double approximate_Hausdorff_distance(const TriangleMesh& tm1,
                                      const TriangleMesh& tm2,
                                      const NamedParameters& np)
{
  return approximate_Hausdorff_distance<Concurrency_tag>(
    tm1, tm2, np, parameters::all_default());
}

template< class Concurrency_tag,
          class TriangleMesh>
double approximate_Hausdorff_distance(const TriangleMesh& tm1,
                                      const TriangleMesh& tm2)
{
  return approximate_Hausdorff_distance<Concurrency_tag>(
    tm1, tm2, parameters::all_default(), parameters::all_default());
}


template< class Concurrency_tag,
          class TriangleMesh,
          class NamedParameters>
double approximate_symmetric_Hausdorff_distance(const TriangleMesh& tm1,
                                                const TriangleMesh& tm2,
                                                const NamedParameters& np)
{
  return approximate_symmetric_Hausdorff_distance<Concurrency_tag>(
    tm1, tm2, np, parameters::all_default());
}

template< class Concurrency_tag,
          class TriangleMesh>
double approximate_symmetric_Hausdorff_distance(const TriangleMesh& tm1,
                                                const TriangleMesh& tm2)
{
  return approximate_symmetric_Hausdorff_distance<Concurrency_tag>(
    tm1, tm2, parameters::all_default(), parameters::all_default());
}

////////////////////////////////////////////////////////////////////////

namespace internal {
/*
#if defined(CGAL_LINKED_WITH_TBB)
template <class AABB_tree, class Point_3>
struct Distance_computation{
  const AABB_tree& tree;
  const std::vector<Point_3>& sample_points;
  Point_3 initial_hint;
  tbb::atomic<double>* distance;

  Distance_computation(
          const AABB_tree& tree,
          const Point_3& p,
          const std::vector<Point_3>& sample_points,
          tbb::atomic<double>* d)
    : tree(tree)
    , sample_points(sample_points)
    , initial_hint(p)
    , distance(d)
  {}

  void
  operator()(const tbb::blocked_range<std::size_t>& range) const
  {
    Point_3 hint = initial_hint;
    double hdist = 0;
    for( std::size_t i = range.begin(); i != range.end(); ++i)
    {
      hint = tree.closest_point(sample_points[i], hint);
      typename Kernel_traits<Point_3>::Kernel::Compute_squared_distance_3 squared_distance;
      double d = to_double(CGAL::approximate_sqrt( squared_distance(hint,sample_points[i]) ));
      if (d>hdist) hdist=d;
    }

    // update max value stored in distance
    double current_value = *distance;
    while( current_value < hdist )
    {
      current_value = distance->compare_and_swap(hdist, current_value);
    }
  }
};
#endif
*/
template <class Concurrency_tag,
          class Kernel,
          class TriangleMesh,
          class VPM1,
          class VPM2>
double bounded_error_Hausdorff_impl(
  const TriangleMesh& tm1,
  const TriangleMesh& tm2,
  const typename Kernel::FT& error_bound,
  VPM1 vpm1,
  VPM2 vpm2)
{
  CGAL_assertion_code(  bool is_triangle = is_triangle_mesh(tm1) && is_triangle_mesh(tm2) );
  CGAL_assertion_msg (is_triangle,
        "One of the meshes is not triangulated. Distance computing impossible.");

  typedef AABB_face_graph_triangle_primitive<TriangleMesh, VPM1> TM1_primitive;
  typedef AABB_face_graph_triangle_primitive<TriangleMesh, VPM2> TM2_primitive;
  typedef AABB_tree< AABB_traits<Kernel, TM1_primitive> > TM1_tree;
  typedef AABB_tree< AABB_traits<Kernel, TM2_primitive> > TM2_tree;
  typedef typename AABB_tree< AABB_traits<Kernel, TM2_primitive> >::AABB_traits Tree_traits;
  typedef typename Tree_traits::Point_and_primitive_id Point_and_primitive_id;

  typedef typename Kernel::Point_3 Point_3;
  typedef typename Kernel::Triangle_3 Triangle_3;

  typedef typename boost::graph_traits<TriangleMesh>::vertex_descriptor vertex_descriptor;
  typedef typename boost::graph_traits<TriangleMesh>::face_descriptor face_descriptor;
  typedef typename boost::graph_traits<TriangleMesh>::halfedge_descriptor halfedge_descriptor;

  typedef std::pair<double, double> Hausdorff_bounds;
  typedef CGAL::Spatial_sort_traits_adapter_3<Kernel,VPM1> Search_traits_3;
  typedef CGAL::dynamic_vertex_property_t<std::pair<double, face_descriptor>> Vertex_property_tag;
  typedef CGAL::dynamic_face_property_t<Hausdorff_bounds> Face_property_tag;

  typedef typename boost::property_map<TriangleMesh, Vertex_property_tag>::const_type Vertex_closest_triangle_map;
  typedef typename boost::property_map<TriangleMesh, Face_property_tag>::const_type Triangle_hausdorff_bounds;

  typedef std::pair<double, double> Hausdorff_bounds;
  typedef std::pair<Triangle_3, Hausdorff_bounds> Candidate_triangle;
  typedef typename std::vector<Candidate_triangle> Candidate_set;

  typename Kernel::Compute_squared_distance_3 squared_distance;
  typename Kernel::Construct_projected_point_3 project_point;
  typename Kernel::FT dist;

  // Build an AABB tree on tm1
  TM1_tree tm1_tree( faces(tm1).begin(), faces(tm1).end(), tm1, vpm1 );
  tm1_tree.build();
  tm1_tree.accelerate_distance_queries();

  // Build an AABB tree on tm2
  TM2_tree tm2_tree( faces(tm2).begin(), faces(tm2).end(), tm2, vpm2 );
  tm2_tree.build();
  tm2_tree.accelerate_distance_queries();
  std::pair<Point_3, face_descriptor> hint = tm2_tree.any_reference_point_and_id();

  // Build traversal traits for tm1_tree
  Hausdorff_primitive_traits_tm1<Tree_traits, Point_3, Kernel, TriangleMesh, VPM1, VPM2> traversal_traits_tm1( tm1_tree.traits(), tm2_tree, tm1, tm2, vpm1, vpm2 );
  // Find candidate triangles in TM1 which might realise the Hausdorff bound
  tm1_tree.traversal( Point_3(0,0,0), traversal_traits_tm1 ); // dummy point given as query as not needed

  // TODO Implement the candidate_triangles set as Stack instead of Vector
  //      check: https://www.boost.org/doc/libs/1_55_0/doc/html/heap.html
  //      Can already build a sorted structure while collecting the candidates
  Candidate_set candidate_triangles = traversal_traits_tm1.get_candidate_triangles();
  Hausdorff_bounds global_bounds = traversal_traits_tm1.get_global_bounds();

  // std::cout << "Culled " << traversal_traits_tm1.get_num_culled_triangles() << " out of " << tm1.num_faces() << std::endl;

/*
  std::cout << "Found " << candidate_triangles.size() << " candidates." << std::endl;
  for (int i=0; i<candidate_triangles.size(); i++) {
    std::cout << "Triangle " << i << " with bounds ("
              << candidate_triangles[i].second.first << ", "
              << candidate_triangles[i].second.second << ")" << std::endl;
  }
*/

  double squared_error_bound = error_bound * error_bound;

  while ( (global_bounds.second - global_bounds.first > error_bound) && candidate_triangles.size() > 0 ) {

    // std::cout << "Current number candidates: " << candidate_triangles.size() << std::endl;
    // std::cout << "Current global bounds: (" << global_bounds.first << ", " << global_bounds.second << ")" << std::endl;

    // Get the first triangle and its Hausdorff bounds from the candidate set
    Candidate_triangle triangle_and_bound = candidate_triangles.back();
    // Remove it from the candidate set as it will be processed now
    candidate_triangles.pop_back();

    // Only process the triangle if it can contribute to the Hausdorff distance,
    // i.e. if its Upper Bound is higher than the currently known best lower bound
    // and the difference between the bounds to be obtained is larger than the
    // user given error.
    Hausdorff_bounds triangle_bounds = triangle_and_bound.second;

    // std::cout << "Current triangle bounds: (" << triangle_bounds.first << ", " << triangle_bounds.second << ")" << std::endl;

    if ( (triangle_bounds.second > global_bounds.first) && (triangle_bounds.second - triangle_bounds.first > error_bound) ) {
      // Get the triangle that is to be subdivided and read its vertices
      Triangle_3 triangle_for_subdivision = triangle_and_bound.first;
      Point_3 v0 = triangle_for_subdivision.vertex(0);
      Point_3 v1 = triangle_for_subdivision.vertex(1);
      Point_3 v2 = triangle_for_subdivision.vertex(2);

      // Check second stopping condition: All three vertices of the triangle
      // are projected onto the same triangle in TM2
      Point_and_primitive_id closest_triangle_v0 = tm2_tree.closest_point_and_primitive(v0);
      Point_and_primitive_id closest_triangle_v1 = tm2_tree.closest_point_and_primitive(v1);
      Point_and_primitive_id closest_triangle_v2 = tm2_tree.closest_point_and_primitive(v2);
      if( (closest_triangle_v0.second == closest_triangle_v1.second) && (closest_triangle_v1.second == closest_triangle_v2.second)) {
        // The upper bound of this triangle is the actual Hausdorff distance of
        // the triangle to the second mesh. Use it as new global lower bound.
        // TODO Update the reference to the realizing triangle here as this is the best current guess.
        global_bounds.first = triangle_bounds.second;
        continue;
      }

      // Check third stopping condition: All edge lengths of the triangle are
      // smaller than the given error bound, cannot get results beyond this
      // bound.
      if (    squared_distance( v0, v1 ) < squared_error_bound
          &&  squared_distance( v0, v2 ) < squared_error_bound
          &&  squared_distance( v1, v2 ) < squared_error_bound ) {
            // The upper bound of this triangle is within error tolerance of
            // the actual upper bound, use it.
            global_bounds.first = triangle_bounds.second;
            continue;
      }

      // Subdivide the triangle into four smaller triangles
      Point_3 v01 = midpoint( v0, v1 );
      Point_3 v02 = midpoint( v0, v2 );
      Point_3 v12 = midpoint( v1, v2 );
      std::array<Triangle_3,4> sub_triangles = {
        Triangle_3( v0, v01, v02), Triangle_3( v1, v01, v12),
        Triangle_3( v2, v02, v12), Triangle_3( v01, v02, v12)
      };

      // Send each of the four triangles to Culling on B with the bounds of the parent triangle
      for (int i=0; i<4; i++) {
        // Call Culling on B with the single triangle found.
        Hausdorff_primitive_traits_tm2<Tree_traits, Triangle_3, Kernel, TriangleMesh, VPM2> traversal_traits_tm2(
          tm2_tree.traits(), tm2, vpm2,
          triangle_bounds.first,
          triangle_bounds.second,
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::infinity()
        );
        tm2_tree.traversal(sub_triangles[i], traversal_traits_tm2);

        // Get the highest current bound from all candidate triangles
        double current_max = 0.;
        for(auto&& ct: candidate_triangles) {
          if (ct.second.second > current_max) {
            current_max = ct.second.second;
          }
        }

        // Update global Hausdorff bounds according to the obtained local bounds
        Hausdorff_bounds local_bounds = traversal_traits_tm2.get_local_bounds();
        if (local_bounds.first > global_bounds.first) {
          global_bounds.first = local_bounds.first;
        }
        global_bounds.second = std::max(
          std::max(current_max, local_bounds.second),
          global_bounds.first
        );

        // TODO Additionally store the face descriptor of the parent from TM1 in the Candidate_triangle.
        // Add the subtriangle to the candidate list
        candidate_triangles.push_back(Candidate_triangle(sub_triangles[i], local_bounds));

        // std::cout << "Split triangle (" << v0 << ", " << v1 << ", " << v2
        //           << ") with bounds: (" << triangle_bounds.first << ", "
        //           << triangle_bounds.second << "), sub-triangle " << i
        //           << " (" << sub_triangles[i].vertex(0) << ", " << sub_triangles[i].vertex(1) << ", " << sub_triangles[i].vertex(2)
        //           << ") has bounds: ("
        //           << local_bounds.first << ", " << local_bounds.second << "), gobal bounds are: ("
        //           << global_bounds.first << ", " << global_bounds.second << ")" << std::endl;
      }
    }
  }

  // Print result found
/*
  std::cout << "Processing candidates finished, found distance (lower, upper): ("
            << global_bounds.first << ", " << global_bounds.second << ")" << std::endl;
*/

  // Return linear interpolation between found lower and upper bound
  return (global_bounds.first + global_bounds.second) / 2.;

#if !defined(CGAL_LINKED_WITH_TBB)
  CGAL_static_assertion_msg (!(boost::is_convertible<Concurrency_tag, Parallel_tag>::value),
                             "Parallel_tag is enabled but TBB is unavailable.");
#else
  // TODO implement parallelized version of the below here.
  // if (boost::is_convertible<Concurrency_tag,Parallel_tag>::value)
  // {
  //   tbb::atomic<double> distance;
  //   distance=0;
  //   Distance_computation<AABBTree, typename Kernel::Point_3> f(tm2_tree
  // , hint, sample_points, &distance);
  //   tbb::parallel_for(tbb::blocked_range<std::size_t>(0, sample_points.size()), f);
  //   return distance;
  // }
  // else
#endif
/*
  {
    // Store all vertices of tm1 in a vector
    std::vector<vertex_descriptor> tm1_vertices;
    tm1_vertices.reserve(num_vertices(tm1));
    tm1_vertices.insert(tm1_vertices.end(),vertices(tm1).begin(),vertices(tm1).end());

    // Sort vertices along a Hilbert curve
    spatial_sort( tm1_vertices.begin(),
                  tm1_vertices.end(),
                  Search_traits_3(vpm1) );

    // For each vertex in tm1, store the distance to the closest triangle of tm2
    Vertex_closest_triangle_map vctm  = get(Vertex_property_tag(), tm1);
    // For each triangle in tm1, sotre its respective local lower and upper bound
    // on the Hausdorff measure
    Triangle_hausdorff_bounds thb = get(Face_property_tag(), tm1);

    // For each vertex in the first mesh, find the closest triangle in the
    // second mesh, store it and also store the distance to this triangle
    // in a dynamic vertex property
    for(vertex_descriptor vd : tm1_vertices)
    {
      // Get the point represented by the vertex
      typename boost::property_traits<VPM1>::reference pt = get(vpm1, vd);
      // Use the AABB tree to find the closest point and face in tm2
      hint = tm2_tree.closest_point_and_primitive(pt, hint);
      // Compute the distance of the point to the closest point in tm2
      dist = squared_distance(hint.first, pt);
      double d = to_double(dist);
      // Store the distance and the closest triangle in the corresponding map
      put(vctm, vd, std::make_pair(d, hint.second));
    }

    // Maps the faces of tm2 to actual triangles
    Triangle_from_face_descriptor_map<TriangleMesh, VPM2> face_to_triangle_map(&tm2, vpm2);
    // Initialize global bounds on the Hausdorff measure
    double h_lower = 0.;
    double h_upper = 0.;
    // Initialize an array of candidate triangles in A to be procesed in the
    // following
    std::vector<face_descriptor> candidate_triangles;

    // For each triangle in the first mesh, initialize its local upper and
    // lower bound and store these in a dynamic face property for furture
    // reference
    for(face_descriptor fd : faces(tm1))
    {
      // Initialize the local bounds for the current face fd
      double h_triangle_lower = 0.;
      double h_triangle_upper = std::numeric_limits<double>::infinity();

      // Create a halfedge descriptor for the current face and store the vertices
      halfedge_descriptor hd = halfedge(fd, tm1);
      std::array<vertex_descriptor,3> face_vertices = {source(hd,tm1), target(hd,tm1), target(next(hd, tm1),tm1)};

      // Get the distance and closest triangle in tm2 for each vertex of fd
      std::array<std::pair<double, face_descriptor>,3> vertex_properties = {
          get(vctm, face_vertices[0]),
          get(vctm, face_vertices[1]),
          get(vctm, face_vertices[2])};

      // Convert the closest faces of tm2 to triangles
      std::array<typename Kernel::Triangle_3,3> triangles_in_B = {
          get(face_to_triangle_map, vertex_properties[0].second),
          get(face_to_triangle_map, vertex_properties[1].second),
          get(face_to_triangle_map, vertex_properties[2].second)};

      for(int i=0; i<3; ++i)
      {
        // Iterate over the vertices by i, the distance to the closest point in
        // tm2 computed above is a lower bound for the local triangle
        h_triangle_lower = (std::max)(h_triangle_lower, vertex_properties[i].first);

        // Iterate over the triangles by i, if the triangles are the same, we do
        // not need to compute the distance, only compute it if the triangles
        // differ
        double face_distance_1 = vertex_properties[i].second==vertex_properties[(i+1)%3].second
                               ? vertex_properties[(i+1)%3].first
                               : squared_distance(project_point(triangles_in_B[i], get(vpm1, face_vertices[(i+1)%3])), get(vpm1, face_vertices[(i+1)%3]));
        double face_distance_2 = vertex_properties[i].second==vertex_properties[(i+2)%3].second
                               ? vertex_properties[(i+2)%3].first
                               : squared_distance(project_point(triangles_in_B[i], get(vpm1, face_vertices[(i+2)%3])), get(vpm1, face_vertices[(i+2)%3]));

        // Update the local lower bound of the triangle
        h_triangle_upper = (std::min)(
          (std::max)(
            (std::max)(face_distance_1, face_distance_2),
            vertex_properties[i].first),
          h_triangle_upper);
      }

      // Store the computed lower and upper bound in a dynamic face property
      put(thb, fd, Hausdorff_bounds(h_triangle_lower, h_triangle_upper));
      h_lower = (std::max)(h_lower, h_triangle_lower);
      h_upper = (std::max)(h_upper, h_triangle_upper);

      // Only process the triangle further if it can still contribute to a
      // Hausdorff distance
      if (h_triangle_upper > h_lower) {

        // TODO culling on B

        candidate_triangles.push_back(fd);
      }
    }



    // TODO Iterate over candidate_triangles and kill those which cannot contribute anymore

    // TODO Send the remaining triangles to the Subdivision

    return (CGAL::approximate_sqrt(h_lower)+CGAL::approximate_sqrt(h_upper))/2.;
  }
*/
}

template <class Point_3,
          class TM2_tree,
          class Kernel>
double recursive_hausdorff_subdivision(
  const Point_3& v0,
  const Point_3& v1,
  const Point_3& v2,
  const TM2_tree& tm2_tree,
  const typename Kernel::FT& squared_error_bound)
{
  // If all edge lengths of the triangle are below the error_bound,
  // return maximum of the distances of the three points to TM2 (via TM2_tree).
  double max_squared_edge_length =
  std::max(
    std::max(
      squared_distance( v0, v1 ),
      squared_distance( v0, v2 )),
    squared_distance( v1, v2 )
  );
  if ( max_squared_edge_length <  squared_error_bound ) {
    return std::max(
      std::max(
        squared_distance( v0, tm2_tree.closest_point(v0) ),
        squared_distance( v1, tm2_tree.closest_point(v1) ) ),
      squared_distance( v2, tm2_tree.closest_point(v2) )
    );
  }

  // Else subdivide the triangle and proceed recursively
  Point_3 v01 = midpoint( v0, v1 );
  Point_3 v02 = midpoint( v0, v2 );
  Point_3 v12 = midpoint( v1, v2 );

  return std::max (
      std::max(
        recursive_hausdorff_subdivision<Point_3, TM2_tree, Kernel>( v0,v01,v02,tm2_tree,squared_error_bound ),
        recursive_hausdorff_subdivision<Point_3, TM2_tree, Kernel>( v1,v01,v12,tm2_tree,squared_error_bound )
      ),
      std::max(
        recursive_hausdorff_subdivision<Point_3, TM2_tree, Kernel>( v2,v02,v12,tm2_tree,squared_error_bound ),
        recursive_hausdorff_subdivision<Point_3, TM2_tree, Kernel>( v01,v02,v12,tm2_tree,squared_error_bound )
      )
    );
}

template <class Concurrency_tag,
          class Kernel,
          class TriangleMesh,
          class VPM1,
          class VPM2>
double bounded_error_Hausdorff_naive_impl(
  const TriangleMesh& tm1,
  const TriangleMesh& tm2,
  const typename Kernel::FT& error_bound,
  VPM1 vpm1,
  VPM2 vpm2)
{
  CGAL_assertion_code(  bool is_triangle = is_triangle_mesh(tm1) && is_triangle_mesh(tm2) );
  CGAL_assertion_msg (is_triangle,
        "One of the meshes is not triangulated. Distance computing impossible.");

  typedef AABB_face_graph_triangle_primitive<TriangleMesh, VPM2> TM2_primitive;
  typedef AABB_tree< AABB_traits<Kernel, TM2_primitive> > TM2_tree;

  typedef typename boost::graph_traits<TriangleMesh>::face_descriptor face_descriptor;

  typedef typename Kernel::Point_3 Point_3;
  typedef typename Kernel::Triangle_3 Triangle_3;

  // Initially, no lower bound is known
  double squared_lower_bound = 0.;
  // Work with squares in the following, only draw sqrt at the very end
  double squared_error_bound = error_bound * error_bound;

  // Build an AABB tree on tm2
  TM2_tree tm2_tree( faces(tm2).begin(), faces(tm2).end(), tm2, vpm2 );
  tm2_tree.build();
  tm2_tree.accelerate_distance_queries();

  // Build a map to obtain actual triangles from the face descriptors of tm1.
  Triangle_from_face_descriptor_map<TriangleMesh, VPM1> face_to_triangle_map( &tm1, vpm1 );

  // Iterate over the triangles of TM1.
  for(face_descriptor fd : faces(tm1))
  {
    // Get the vertices of the face and pass them on to a recursive method.
    Triangle_3 triangle = get(face_to_triangle_map, fd);
    Point_3 v0 = triangle.vertex(0);
    Point_3 v1 = triangle.vertex(1);
    Point_3 v2 = triangle.vertex(2);

    // Recursively process the current triangle to obtain a lower bound on
    // its Hausdorff distance.
    double triangle_bound = recursive_hausdorff_subdivision<Point_3, TM2_tree, Kernel>( v0, v1, v2, tm2_tree, squared_error_bound );

    // Store the largest lower bound.
    if( triangle_bound > squared_lower_bound ) {
      squared_lower_bound = triangle_bound;
    }
  }

  // Return linear interpolation between found upper and lower bound
  return (approximate_sqrt( squared_lower_bound ));

#if !defined(CGAL_LINKED_WITH_TBB)
  CGAL_static_assertion_msg (!(boost::is_convertible<Concurrency_tag, Parallel_tag>::value),
                             "Parallel_tag is enabled but TBB is unavailable.");
#else
  // TODO implement parallelized version of the below here.
#endif
}

} //end of namespace internal

/**
 * \ingroup PMP_distance_grp
 * computes the approximate Hausdorff distance from `tm1` to `tm2` by returning
 * the distance of the farthest point from `tm2` amongst a sampling of `tm1`
 * generated with the function `sample_triangle_mesh()` with
 * `tm1` and `np1` as parameter.
 *
 * A parallel version is provided and requires the executable to be
 * linked against the <a href="https://www.threadingbuildingblocks.org">Intel TBB library</a>.
 * To control the number of threads used, the user may use the `tbb::task_scheduler_init` class.
 * See the <a href="https://www.threadingbuildingblocks.org/documentation">TBB documentation</a>
 * for more details.
 *
 * @tparam Concurrency_tag enables sequential versus parallel algorithm.
 *                         Possible values are `Sequential_tag`
 *                         and `Parallel_tag`.
 * @tparam TriangleMesh a model of the concept `FaceListGraph`
 * @tparam NamedParameters1 a sequence of \ref pmp_namedparameters "Named Parameters" for `tm1`
 * @tparam NamedParameters2 a sequence of \ref pmp_namedparameters "Named Parameters" for `tm2`
 *
 * @param tm1 the triangle mesh that will be sampled
 * @param tm2 the triangle mesh to compute the distance to
 * @param np1 optional sequence of \ref pmp_namedparameters "Named Parameters" for `tm1` passed to `sample_triangle_mesh()`.
 *
 * @param np2 optional sequence of \ref pmp_namedparameters "Named Parameters" for `tm2` among the ones listed below
 *
 * \cgalNamedParamsBegin
 *    \cgalParamBegin{vertex_point_map} the property map with the points associated to the vertices of `tm2`
 *      If this parameter is omitted, an internal property map for `CGAL::vertex_point_t` must be available in `TriangleMesh`
 *      and in all places where `vertex_point_map` is used.
 *    \cgalParamEnd
 * \cgalNamedParamsEnd
 * The function `CGAL::parameters::all_default()` can be used to indicate to use the default values for
 * `np1` and specify custom values for `np2`
 */

/*
 * Implementation of Bounded Hausdorff distance computation using AABBTree
 * culling.
 */
template< class Concurrency_tag,
          class TriangleMesh,
          class NamedParameters1,
          class NamedParameters2>
double bounded_error_Hausdorff_distance( const TriangleMesh& tm1,
                                         const TriangleMesh& tm2,
                                         double error_bound,
                                         const NamedParameters1& np1,
                                         const NamedParameters2& np2)
{
  typedef typename GetGeomTraits<TriangleMesh,
                                 NamedParameters1>::type Geom_traits;

   typedef typename GetVertexPointMap<TriangleMesh, NamedParameters1>::const_type Vpm1;
   typedef typename GetVertexPointMap<TriangleMesh, NamedParameters2>::const_type Vpm2;

   using boost::choose_param;
   using boost::get_param;

   Vpm1 vpm1 = choose_param(get_param(np1, internal_np::vertex_point),
                           get_const_property_map(vertex_point, tm1));
   Vpm2 vpm2 = choose_param(get_param(np2, internal_np::vertex_point),
                           get_const_property_map(vertex_point, tm2));

   return internal::bounded_error_Hausdorff_impl<Concurrency_tag, Geom_traits>(tm1, tm2, error_bound, vpm1, vpm2);
}

template< class Concurrency_tag,
          class TriangleMesh,
          class NamedParameters1>
double bounded_error_Hausdorff_distance( const TriangleMesh& tm1,
                                         const TriangleMesh& tm2,
                                         double error_bound,
                                         const NamedParameters1& np1)
{
  return bounded_error_Hausdorff_distance<Concurrency_tag>(tm1, tm2, error_bound, np1, parameters::all_default());
}

template< class Concurrency_tag,
          class TriangleMesh>
double bounded_error_Hausdorff_distance( const TriangleMesh& tm1,
                                         const TriangleMesh& tm2,
                                         double error_bound)
{
  return bounded_error_Hausdorff_distance<Concurrency_tag>(tm1, tm2, error_bound, parameters::all_default() );
}

/*
 * Implementation of naive Bounded Hausdorff distance computation.
 */
template< class Concurrency_tag,
          class TriangleMesh,
          class NamedParameters1,
          class NamedParameters2>
double bounded_error_Hausdorff_distance_naive( const TriangleMesh& tm1,
                                         const TriangleMesh& tm2,
                                         double error_bound,
                                         const NamedParameters1& np1,
                                         const NamedParameters2& np2)
{
  typedef typename GetGeomTraits<TriangleMesh,
                                 NamedParameters1>::type Geom_traits;

   typedef typename GetVertexPointMap<TriangleMesh, NamedParameters1>::const_type Vpm1;
   typedef typename GetVertexPointMap<TriangleMesh, NamedParameters2>::const_type Vpm2;

   using boost::choose_param;
   using boost::get_param;

   Vpm1 vpm1 = choose_param(get_param(np1, internal_np::vertex_point),
                           get_const_property_map(vertex_point, tm1));
   Vpm2 vpm2 = choose_param(get_param(np2, internal_np::vertex_point),
                           get_const_property_map(vertex_point, tm2));

   return internal::bounded_error_Hausdorff_naive_impl<Concurrency_tag, Geom_traits>(tm1, tm2, error_bound, vpm1, vpm2);
}

template< class Concurrency_tag,
          class TriangleMesh,
          class NamedParameters1>
double bounded_error_Hausdorff_distance_naive( const TriangleMesh& tm1,
                                         const TriangleMesh& tm2,
                                         double error_bound,
                                         const NamedParameters1& np1)
{
  return bounded_error_Hausdorff_distance_naive<Concurrency_tag>(tm1, tm2, error_bound, np1, parameters::all_default());
}

template< class Concurrency_tag,
          class TriangleMesh>
double bounded_error_Hausdorff_distance_naive( const TriangleMesh& tm1,
                                         const TriangleMesh& tm2,
                                         double error_bound)
{
  return bounded_error_Hausdorff_distance_naive<Concurrency_tag>(tm1, tm2, error_bound, parameters::all_default() );
}

} } // end of namespace CGAL::Polygon_mesh_processing


#endif //CGAL_POLYGON_MESH_PROCESSING_DISTANCE_H
