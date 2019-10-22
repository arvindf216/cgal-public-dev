// Copyright (c) 2010 GeometryFactory (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org); you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 3 of the License,
// or (at your option) any later version.
//
// Licensees holding a valid commercial license may use this file in
// accordance with the commercial license agreement provided with the software.
//
// This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
// WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
//
// $URL$
// $Id$
// SPDX-License-Identifier: LGPL-3.0+
//
//
// Author(s)     : Sebastien Loriot
//

#ifndef CGAL_INTERSECTIONS_3_BBOX_3_BBOX_3_H
#define CGAL_INTERSECTIONS_3_BBOX_3_BBOX_3_H

#include <CGAL/Bbox_3.h>
#include <CGAL/Intersection_traits_3.h>

namespace CGAL {

bool
inline
do_intersect(const CGAL::Bbox_3& c,
             const CGAL::Bbox_3& bbox)
{
  return CGAL::do_overlap(c, bbox);
}

typename boost::optional< typename boost::variant< Bbox_3> >
inline
intersection(const CGAL::Bbox_3& a,
             const CGAL::Bbox_3& b)
{
  typedef typename boost::variant<Bbox_3> variant_type;
  typedef typename boost::optional<variant_type> Result_type;

  if(!do_intersect(a,b))
    return Result_type();

  double xmin = (std::max)(a.xmin(), b.xmin());
  double xmax = (std::min)(a.xmax(), b.xmax());
  double ymin = (std::max)(a.ymin(), b.ymin());
  double ymax = (std::min)(a.ymax(), b.ymax());
  double zmin = (std::max)(a.zmin(), b.zmin());
  double zmax = (std::min)(a.zmax(), b.zmax());

  return Result_type(std::forward<Bbox_3>(Bbox_3(xmin, ymin, zmin, xmax, ymax, zmax)));
}

} //namespace CGAL


#endif // CGAL_INTERSECTIONS_3_BBOX_3_BBOX_3_H
