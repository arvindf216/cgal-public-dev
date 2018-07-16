// Copyright (c) 2014 INRIA Sophia-Antipolis (France).
// All rights reserved.
//
// This file is a part of CGAL (www.cgal.org).
// You can redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $URL$
// $Id$
// SPDX-License-Identifier: GPL-3.0+
//
// Author(s) : Keyu Chen， Dmitry Anisimov.

/*!
  \file Maximum_entropy_2.h
*/

#ifndef CGAL_MAXIMUM_ENTROPY_2_H
#define CGAL_MAXIMUM_ENTROPY_2_H

#include <CGAL/license/Barycentric_coordinates_2.h>

#include <CGAL/disable_warnings.h>

// CGAL headers.
#include <CGAL/assertions.h>
#include <CGAL/Polygon_2_algorithms.h>

// Barycentric coordinates headers.
#include <CGAL/Barycentric_coordinates_2/barycentric_enum_2.h>

// Boost headers.
#include <boost/optional/optional.hpp>

// Eigen headers.
#include <CGAL/Eigen_vector.h>
#include <CGAL/Eigen_matrix.h>

// Add partition headers.
#include <CGAL/Barycentric_coordinates_2/Maximum_entropy_2/Partition.h>



// CGAL namespace.
namespace CGAL {

// Barycentric coordinates namespace.
namespace Barycentric_coordinates {

// Brief introduction about Maximum Entropy coordinates




template<class Traits, class Prior, class Solver >
    class Maximum_entropy_2
{

public:

    /// \name Types
    /// @{

    /// Number type.
    typedef typename Traits::FT      FT;

    /// Point type.
    typedef typename Traits::Point_2 Point_2;

    /// @}



    // \name Creation

    // Brief introduction of Maximum_entropy_2 class, its constructor, input and output etc.
    Maximum_entropy_2(const std::vector<typename Traits::Point_2> &vertices, const Traits &barycentric_traits) :
        m_vertex(vertices),
        m_barycentric_traits(barycentric_traits),
        prior(Prior(m_vertex, m_barycentric_traits)),
        solver(Solver(m_vertex, m_barycentric_traits)),
        partition(Partition())
    {
        // Initialize some private parameters here.
        const size_t number_of_vertices = m_vertex.size();
        m.resize(number_of_vertices);
        z.resize(number_of_vertices);
    }

    // Computation of Maximum Entropy Weight Functions, to keep this interface the same with other coordinates.

    // This function computes weights for single query point, but in this case, Maximum Entropy coordinate can
    // not provide such weights.
    // We keep this interface, leave the content empty except an assertion.
    template<class OutputIterator>
        inline boost::optional<OutputIterator> weights(const Point_2 &query_point, OutputIterator &output)
    {

    }

    // Computation of Maximum Entropy Basis Functions

    // This function computes Maximum Entropy barycentric coordinates for a chosen query point on the bounded side of an arbitrary polygon.
    // \pre The provided polygon is arbitrary one.
    template<class OutputIterator>
        inline boost::optional<OutputIterator> coordinates_on_bounded_side(const Point_2 &query_point, OutputIterator &output, const Type_of_algorithm type_of_algorithm)
    {
        switch(type_of_algorithm)
        {
            case PRECISE:
            return coordinates_on_bounded_side_precise_2(query_point, output);

            case FAST:
            return coordinates_on_bounded_side_fast_2(query_point, output);

            default:
            break;
        }

        // Pointer cannot be here. Something went wrong.
        const bool type_of_algorithm_failure = true;
        if(!type_of_algorithm_failure) return boost::optional<OutputIterator>(output);
        else return boost::optional<OutputIterator>();
    }

    // This function computes Maximum Entropy barycentric coordinates for a chosen query point on the unbounded side of an arbitrary polygon.
    // Due to the constraint of Maximum Entropy coordinate, we can not compute coordinates for unbounded side points.
    // We keep the interface and leave this function empty except an assertion.
    template<class OutputIterator>
        inline boost::optional<OutputIterator> coordinates_on_unbounded_side(const Point_2 &query_point, OutputIterator &output, const Type_of_algorithm type_of_algorithm, const bool warning_tag = true)
    {

    }

  

private:

    // Some convenient typedefs.
    typedef typename Traits::Vector_2 Vector_2;
    typedef typename std::vector<FT>      FT_vector;
    typedef typename std::vector<Point_2> Point_vector;
    typedef typename CGAL::Eigen_matrix<FT>        Matrix;

    // Internal global variables.
    const Point_vector &m_vertex;

    const Traits &m_barycentric_traits;

    // Prior class
    Prior prior;

    // Solver class
    Solver solver;

    // Partition class
    Partition partition;

    FT_vector m, z;

    template<class OutputIterator>
        boost::optional<OutputIterator> coordinates_on_bounded_side_precise_2(const Point_2 &query_point, OutputIterator &output)
    {
        // Implementation of precise mec computing.
        Vector_2 s;
        const size_t number_of_vertices = m_vertex.size();

        Matrix vtilde(number_of_vertices, 2);

        for(size_t i = 0; i < number_of_vertices; ++i) {
            s = Vector_2(m_vertex[i], query_point);

            vtilde.set(i, 0, s.x());
            vtilde.set(i, 1, s.y());
        }

        prior.compute(query_point, m);

        FT_vector lambda(2);
        solver.solve(lambda, vtilde, m, PRECISE);

        FT Z(0);
        for(size_t i = 0; i < number_of_vertices; ++i) {
            z[i] = partition(vtilde, m, lambda, i);
            Z += z[i];
        }

        for(size_t i = 0; i < number_of_vertices; ++i) {
            *output = z[i] / Z;
            output++;
        }

    }

    template<class OutputIterator>
        boost::optional<OutputIterator> coordinates_on_bounded_side_fast_2(const Point_2 &query_point, OutputIterator &output)
    {
        // Implementation of fast mec computing.
        Vector_2 s;
        const size_t number_of_vertices = m_vertex.size();

        Matrix vtilde(number_of_vertices, 2);

        for(size_t i = 0; i < number_of_vertices; ++i) {
            s = Vector_2(m_vertex[i], query_point);

            vtilde.set(i, 0, s.x());
            vtilde.set(i, 1, s.y());
        }


        prior.compute(query_point, m);


        FT_vector lambda(2);
        solver.solve(lambda, vtilde, m, FAST);

        FT Z(0);
        for(size_t i = 0; i < number_of_vertices; ++i) {
            z[i] = partition(vtilde, m, lambda, (int) i);
            Z += z[i];
        }

        for(size_t i = 0; i < number_of_vertices; ++i) {
            *output = z[i] / Z;
            output++;
        }

        return boost::optional<OutputIterator>(output);

    }

};

} // namespace Barycentric_coordinates

} // namespace CGAL

#include <CGAL/enable_warnings.h>

#endif // CGAL_MAXIMUM_ENTROPY_2_H