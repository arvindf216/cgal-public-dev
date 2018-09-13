#ifndef CGAL_LEVEL_OF_DETAIL_CONSTRAINED_TRIANGULATION_CREATOR_H
#define CGAL_LEVEL_OF_DETAIL_CONSTRAINED_TRIANGULATION_CREATOR_H

// STL includes.
#include <vector>

// LOD includes.
#include <CGAL/Level_of_detail/internal/utils.h>

namespace CGAL {

namespace Level_of_detail {

template<class InputKernel, class InputTriangulation>
class Constrained_triangulation_creator {

public:
  using Kernel        = InputKernel;
  using Triangulation = InputTriangulation;

  using Point_2 = typename Kernel::Point_2;
  using FT = typename Kernel::FT;

  using Triangulation_vertex_handle  = typename Triangulation::Vertex_handle;
  using Triangulation_vertex_handles = std::vector< std::vector<Triangulation_vertex_handle> >;
  using Triangulation_faces_iterator = typename Triangulation::Finite_faces_iterator;

  template<class BBox, class Faces_range>
  void make_triangulation_with_info(const BBox& bbox,
                                    FT bbox_step,
                                    const Faces_range &faces_range,
                                    Triangulation &triangulation) const {

    triangulation.clear();

    // Insert bbox refined
    FT dx = bbox[2].x() - bbox[0].x();
    FT dy = bbox[2].y() - bbox[0].y();
    std::size_t nb_x = (dx / bbox_step) + 1;
    std::size_t nb_y = (dy / bbox_step) + 1;

    typename Triangulation::Face_handle hint;
    for (std::size_t i = 0; i <= nb_x; ++ i)
    {
      FT x = bbox[0].x() + dx * (i / FT(nb_x));
      for (std::size_t j = 0; j <= nb_y; ++ j)
      {
        FT y = bbox[0].y() + dy * (j / FT(nb_y));
        Point_2 p(x,y);
        hint = triangulation.locate(p,hint);
        triangulation.insert(Point_2(x, y), hint);
      }
    }
    // Insert points.
    Triangulation_vertex_handles triangulation_vertex_handles;
    triangulation_vertex_handles.resize(faces_range.size());
    insert_points(faces_range, triangulation, triangulation_vertex_handles);

    // Insert constraints.
    insert_constraints(triangulation_vertex_handles, triangulation);
                
    // Update info.
    update_info(faces_range, triangulation);
        
  }

private:
  template<class Input_faces_range>
  void insert_points(const Input_faces_range &input_faces_range, Triangulation &triangulation, Triangulation_vertex_handles &triangulation_vertex_handles) const {

    using Input_faces_iterator = typename Input_faces_range::const_iterator;

    size_t i = 0;
    typename Triangulation::Face_handle hint;
    
    for (Input_faces_iterator if_it = input_faces_range.begin(); if_it != input_faces_range.end(); ++if_it, ++i) {
					
      if (if_it->visibility_label() == Visibility_label::OUTSIDE)
        continue;

      bool okay = true;
      const auto &vertices = *if_it;
      for (auto cv_it = vertices.begin(); cv_it != vertices.end(); ++cv_it)
      {
        const Point_2 &point = *cv_it;
        hint = triangulation.locate(point, hint);
        if (triangulation.is_infinite(hint))
        {
          okay = false;
          break;
        }
      }

      if (!okay)
        continue;
      
      triangulation_vertex_handles[i].resize(vertices.size());

      size_t j = 0;
      for (auto cv_it = vertices.begin(); cv_it != vertices.end(); ++cv_it, ++j) {
						
        const Point_2 &point = *cv_it;
        triangulation_vertex_handles[i][j] = triangulation.insert(point, hint);
      }
    }
  }

  void insert_constraints(const Triangulation_vertex_handles &triangulation_vertex_handles, Triangulation &triangulation) const {
                
    const size_t size_i = triangulation_vertex_handles.size();
    for (size_t i = 0; i < size_i; ++i) {

      const size_t size_j = triangulation_vertex_handles[i].size();
      for (size_t j = 0; j < size_j; ++j) {
        const size_t jp = (j + 1) % size_j;
						
        if (triangulation_vertex_handles[i][j] != triangulation_vertex_handles[i][jp])
          triangulation.insert_constraint(triangulation_vertex_handles[i][j], triangulation_vertex_handles[i][jp]);
      }
    }
  }

  template<class Input_faces_range>
  void update_info(const Input_faces_range &input_faces_range, Triangulation &triangulation) const {
    using Input_faces_iterator = typename Input_faces_range::const_iterator;
                
    for (Triangulation_faces_iterator tf_it = triangulation.finite_faces_begin(); tf_it != triangulation.finite_faces_end(); ++tf_it) {
      Point_2 barycentre = internal::barycenter<Kernel> (tf_it);
  
      for (Input_faces_iterator if_it = input_faces_range.begin(); if_it != input_faces_range.end(); ++if_it) {
        const auto &polygon = *if_it;

        if (polygon.visibility_label() != Visibility_label::OUTSIDE &&
            polygon.has_on_bounded_side(barycentre))
        {
          tf_it->info().visibility_label() = polygon.visibility_label();
          break;
        }
      }
    }
  }
};

} // Level_of_detail

} // CGAL

#endif // CGAL_LEVEL_OF_DETAIL_CONSTRAINED_TRIANGULATION_CREATOR_H