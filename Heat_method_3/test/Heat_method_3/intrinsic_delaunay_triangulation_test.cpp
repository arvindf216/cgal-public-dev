#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Dynamic_property_map.h>
#include <CGAL/Heat_method_3/Intrinsic_Delaunay_Triangulation_3.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <Eigen/Sparse>
#include <Eigen/Dense>

typedef CGAL::Simple_cartesian<double>                       Kernel;
typedef Kernel::Point_3                                      Point;
typedef Kernel::Point_2                                      Point_2;
typedef CGAL::Surface_mesh<Point>                            Mesh;
//typedef CGAL::Polyhedron_3<Kernel> Mesh;

typedef CGAL::dynamic_vertex_property_t<double> Vertex_distance_tag;
typedef boost::property_map<Mesh, Vertex_distance_tag >::type Vertex_distance_map;

typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;

typedef CGAL::dynamic_halfedge_property_t<Point_2> Halfedge_coordinate_tag;
typedef boost::property_map<Mesh, Halfedge_coordinate_tag >::type Halfedge_coordinate_map;


typedef CGAL::Intrinsic_Delaunay_Triangulation_3::Intrinsic_Delaunay_Triangulation_3<Mesh,Kernel,Vertex_distance_map, Halfedge_coordinate_map> IDT;


int main()
{
  Mesh sm;
  Vertex_distance_map vertex_distance_map = get(Vertex_distance_tag(),sm);
  Halfedge_coordinate_map halfedge_coord_map = get(Halfedge_coordinate_tag(), sm);

  std::ifstream in("data/brain100k.off");
  in >> sm;
  if(!in || num_vertices(sm) == 0) {
    std::cerr << "Problem loading the input data" << std::endl;
    return 1;
  }
  int a = num_faces(sm);
  IDT im(sm, vertex_distance_map, halfedge_coord_map);

  std::cout<<"success \n";
  return 0;

}