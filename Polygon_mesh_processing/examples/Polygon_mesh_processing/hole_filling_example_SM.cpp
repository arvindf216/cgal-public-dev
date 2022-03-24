#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <set>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3                                     Point;
typedef CGAL::Surface_mesh<Point>                           Mesh;

typedef boost::graph_traits<Mesh>::vertex_descriptor        vertex_descriptor;
typedef boost::graph_traits<Mesh>::halfedge_descriptor      halfedge_descriptor;
typedef boost::graph_traits<Mesh>::face_descriptor          face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;

bool is_small_hole(halfedge_descriptor h, Mesh & mesh,
                   double max_hole_diam, int max_num_hole_edges)
{
  int num_hole_edges = 0;
  CGAL::Bbox_3 hole_bbox;
  for (halfedge_descriptor hc : CGAL::halfedges_around_face(h, mesh))
  {
    const Point& p = mesh.point(target(hc, mesh));

    hole_bbox += p.bbox();
    ++num_hole_edges;

    // Exit early, to avoid unnecessary traversal of large holes
    if (num_hole_edges > max_num_hole_edges) return false;
    if (hole_bbox.xmax() - hole_bbox.xmin() > max_hole_diam) return false;
    if (hole_bbox.ymax() - hole_bbox.ymin() > max_hole_diam) return false;
    if (hole_bbox.zmax() - hole_bbox.zmin() > max_hole_diam) return false;
  }

  return true;
}

struct Progress {

    Progress()
    {}

    Progress(const Progress&) = delete;

    void start_delaunay_triangulation() const
    {
        std::cout << "Start Delaunay phase" << std::endl;
    }

    void end_delaunay_triangulation(bool success) const
    {
        std::cout << "End Delaunay phase " << ((success)?" (success)":" (failed)") << std::endl;

    }

    void start_cubic_algorithm( int n)
    {
        cubic_n = n;
        cubic_i = n / 10;
        std::cout << "Start Cubic phase with " << n << " steps" << std::endl;
    }


    void cubic_algorithm(int i)
    {
        if (i == cubic_i) {
          std::cout << double(cubic_i) / double(cubic_n) * 100 << "%" << std::endl;
          cubic_i += cubic_n / 10;
        }
    }

    void end_cubic_algorithm() const
    {
        std::cout << "End Cubic phase" << std::endl;
    }

    int cubic_n = 0, cubic_i = 0;
};

// Incrementally fill the holes that are no larger than given diameter
// and with no more than a given number of edges (if specified).

int main(int argc, char* argv[])
{
  const std::string filename = (argc > 1) ? argv[1] : CGAL::data_file_path("meshes/mech-holes-shark.off");

  Mesh mesh;
  if(!PMP::IO::read_polygon_mesh(filename, mesh))
  {
    std::cerr << "Invalid input." << std::endl;
    return 1;
  }

  // Both of these must be positive in order to be considered
  double max_hole_diam   = (argc > 2) ? boost::lexical_cast<double>(argv[2]): -1.0;
  int max_num_hole_edges = (argc > 3) ? boost::lexical_cast<int>(argv[3]) : -1;

  unsigned int nb_holes = 0;
  std::vector<halfedge_descriptor> border_cycles;

  // collect one halfedge per boundary cycle
  PMP::extract_boundary_cycles(mesh, std::back_inserter(border_cycles));

  for(halfedge_descriptor h : border_cycles)
  {
    if(max_hole_diam > 0 && max_num_hole_edges > 0 &&
       !is_small_hole(h, mesh, max_hole_diam, max_num_hole_edges))
      continue;

    std::vector<face_descriptor>  patch_facets;
    std::vector<vertex_descriptor> patch_vertices;
    Progress progress;
    bool success = std::get<0>(PMP::triangulate_refine_and_fair_hole(mesh,
                                                                     h,
                                                                     std::back_inserter(patch_facets),
                                                                     std::back_inserter(patch_vertices),
                                                                     PMP::parameters::visitor(std::ref(progress)).use_delaunay_triangulation(false)));

    std::cout << "* Number of facets in constructed patch: " << patch_facets.size() << std::endl;
    std::cout << "  Number of vertices in constructed patch: " << patch_vertices.size() << std::endl;
    std::cout << "  Is fairing successful: " << success << std::endl;
    ++nb_holes;
  }

  std::cout << std::endl;
  std::cout << nb_holes << " holes have been filled" << std::endl;

  CGAL::IO::write_polygon_mesh("filled_SM.off", mesh, CGAL::parameters::stream_precision(17));
  std::cout << "Mesh written to: filled_SM.off" << std::endl;

  return 0;
}
